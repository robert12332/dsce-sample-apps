import os, base64, bcrypt, decimal, hashlib, json, requests, time, uuid, wget
from flask import Flask, request, render_template, Response

import ibmdata.isdw
import ibmdata.isdwtest
import ibmdata.qdat
import pandas as pd
import mysql.connector
from mysql.connector import pooling
from sql_formatter.core import format_sql


from ibm_watson_machine_learning.foundation_models import Model
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredCSVLoader, UnstructuredPDFLoader


from dotenv import load_dotenv

load_dotenv()

WML_SERVER_URL = os.getenv(
    "WML_SERVER_URL", default="https://us-south.ml.cloud.ibm.com"
)
SERVER_URL = os.getenv("SERVER_URL")
FOUNDATION_MODELS_URL = os.getenv("FOUNDATION_MODELS_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
API_KEY = os.getenv("WATSONX_API_KEY", default="")
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": "Bearer {}".format(API_KEY),
}
APIAUTHCODE = os.getenv("APIAUTHCODE", default="wv29efrtsx95")

# env variable for watson assistant
WAINTEGRATIONID = os.getenv("WAINTEGRATIONID")
WAREGION = os.getenv("WAREGION")
WASERVICEINSTANCEID = os.getenv("WASERVICEINSTANCEID")

project_id = WATSONX_PROJECT_ID
model_id = "ibm/granite-13b-instruct-v2"
space_id = None  # optional
verify = False
embedding_model = "sentence-transformers/all-minilm-l6-v2"
credentials = Credentials(url=WML_SERVER_URL, api_key=API_KEY)
client = APIClient(credentials)

params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
}

# per tenant data related to the vector index retrieval
stores = {}  # vector index
retrievers = {}
chains = {}
chat_sessions = {}
users_data = {}
text_converted = {}
# the time when a vector store with a specific id was loaded so that it can cleared after a time limit
load_times = {}
current_label = {}
file_type = {}

users_data = {}
all_prompts = ["rag"]

for i in all_prompts:
    with open("prompts/{}_prompt.txt".format(i), "r") as sample_prompt_f:
        users_data["default_{}_prompt".format(i)] = sample_prompt_f.read()


def setDefaultPrompts(id):
    if id not in users_data:
        users_data[id] = dict(rag={"prompt": users_data["default_rag_prompt"]})
    dict_keys = users_data[id].keys()
    for ai_task in dict_keys:
        with open("payload/{}.json".format(ai_task)) as payload_f:
            payload_f_json = json.load(payload_f)
        users_data[id][ai_task]["model_id"] = payload_f_json["model_id"]
        users_data[id][ai_task]["max_new_tokens"] = payload_f_json["parameters"][
            "max_new_tokens"
        ]
        users_data[id][ai_task]["stop_sequences"] = payload_f_json["parameters"][
            "stop_sequences"
        ]


filename = "state_of_the_union.txt"
url = "https://raw.github.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt"

if not os.path.isfile(filename):
    wget.download(url, out=filename)

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


embeddings = WatsonxEmbeddings(
    model_id=embedding_model,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
)

docsearch = Chroma.from_documents(texts, embeddings)

watsonx_granite = WatsonxLLM(
    model_id=model_id,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
    params=params,
)

app = Flask(__name__)


@app.route('/load/<id>', methods=["POST"])
def load_files(id):
    all_docs = []
    file = request.files["file1"]
    split_tup = os.path.splitext(file.filename)
    unique_filename = uuid.uuid4().hex
    file_path = "upload/{}{}".format(unique_filename, split_tup[1])
    print(file_path)
    file.save(file_path)


@app.route("/")
def serve_index_page():
    return render_template(
        "index.html",
        wa_integration_id=WAINTEGRATIONID,
        wa_region=WAREGION,
        wa_service_instance_id=WASERVICEINSTANCEID,
        api_a_code=APIAUTHCODE,
    )


@app.route("/get-prompt/<id>/<ai_task>")
def get_prompt(id, ai_task):
    try:
        type = current_label[id] if ai_task == "none" else ai_task
        # return {"data":users_data[id][type],"type": type, "ok":True}
        return {
            "prompt": users_data[id][type]["prompt"],
            "model_id": users_data[id][type]["model_id"],
            "max_new_tokens": users_data[id][type]["max_new_tokens"],
            "stop_sequences": users_data[id][type]["stop_sequences"],
            "type": type,
            "ok": True,
        }
    except:
        return {"data": "No data available", "type": type, "ok": True}


@app.route("/verify_user", methods=["POST"])
def verify_user():
    data = request.json
    email = data["email"]

    # User verification logic - Start

    verified = True  # True - if verification is passed, False - if verification failed

    # User verification logic - End

    id = email.split("@")[0]
    if verified:
        setDefaultPrompts(id)
    uploaded_file_types = file_type.get(id)
    response_data = {
        "status": "verified" if verified else "not verified",
        "doc_status": (
            "_available"
            if uploaded_file_types and len(uploaded_file_types) > 0
            else "not_available"
        ),
        "file_types": uploaded_file_types,
    }
    return response_data


qa = RetrievalQA.from_chain_type(
    llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever()
)

query = "What did the president say about Ketanji Brown Jackson"
print(qa.invoke(query))


if __name__ == "__main__":
    SERVICE_PORT = os.getenv("SERVICE_PORT", default="8050")
    DEBUG_MODE = eval(os.getenv("DEBUG_MODE", default="True"))
    app.run(port=SERVICE_PORT, debug=DEBUG_MODE)
# model = ModelInference(
#     model_id=model_id,
#     api_client=client,
#     params=params,
#     project_id=project_id,
#     space_id=space_id,
#     verify=verify,
# )


# def generate_embeddings(embedding_model, ext, input_dir):
#     embed_model = ModelInference(
#         client=client,
#         model_id=embedding_model,
#         parameters=TextEmbeddingParameters(truncate_input_tokens=True),
#     )
#     # load data
#     loader = SimpleDirectoryReader(
#         input_dir=input_dir, required_exts=[ext], recursive=True
#     )
#     docs = loader.load_data()

#     # Creating an index over loaded data
#     Settings.embed_model = embed_model
#     index = VectorStoreIndex.from_documents(docs, show_progress=True)

#     Settings.llm = llm

#     query_engine = index.as_query_engine(streaming=True)

#     return query_engine


# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {
#         "role": "user",
#         "content": [{"type": "text", "text": "How far is Paris from Bangalore?"}],
#     },
#     {
#         "role": "assistant",
#         "content": "The distance between Paris, France, and Bangalore, India, is approximately 7,800 kilometers (4,850 miles)",
#     },
#     {
#         "role": "user",
#         "content": [{"type": "text", "text": "What is the flight distance?"}],
#     },
# ]

# print(model.chat(messages=messages))
