import base64
import datetime
import hashlib
import json
import os
import traceback
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

import operations_images_jpeg_png as img_ops

# One time neo4j index creation:
# CREATE INDEX file_name_index IF NOT EXISTS FOR (f:File) ON (f.name);
# CREATE INDEX file_date_index IF NOT EXISTS FOR (f:File) ON (f.date);
# CREATE INDEX file_timestamp_index IF NOT EXISTS FOR (f:File) ON (f.timestamp);
# CREATE INDEX chunk_origin_filename_index IF NOT EXISTS FOR (c:Chunk) ON (c.origin_filename);
# CREATE INDEX chunk_page_index IF NOT EXISTS FOR (c:Chunk) ON (c.page);

# CREATE INDEX chat_username_index IF NOT EXISTS FOR (c:Chat) ON (c.username);
# CREATE INDEX chat_timestamp_index IF NOT EXISTS FOR (c:Chat) ON (c.timestamp);

NEO4J_URI = st.secrets['NEO4J_URI']
NEO4J_USER = st.secrets['NEO4J_USER']
NEO4J_PASSWORD = st.secrets['NEO4J_PASSWORD']
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

AZURE_OPENAI_MODEL = st.secrets['AZURE_OPENAI_MODEL']
AZURE_OPENAI_ENDPOINT = st.secrets['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_KEY = st.secrets['AZURE_OPENAI_KEY']
AZURE_OPENAI_VERSION = st.secrets['AZURE_OPENAI_VERSION']

AZURE_EMBEDDING_MODEL = st.secrets['AZURE_EMBEDDING_MODEL']
AZURE_EMBEDDING_ENDPOINT = st.secrets['AZURE_EMBEDDING_ENDPOINT']
AZURE_EMBEDDING_KEY = st.secrets['AZURE_EMBEDDING_KEY']

def create_file_and_chunks(file, username, full_text, split_documents, file_abs_path=None, image_data=None,
                           delete_chunks=True):
    graph = Neo4jGraph()
    current_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    file_name = file.split('\\')[-1] if '\\' in file else file.split('/')[-1]

    # Initialize the OpenAI LLM with Azure configuration
    model = AzureChatOpenAI(
        model=AZURE_OPENAI_MODEL,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )
    summary_prompt = ("You are given a text below from a file, summarise from the content and get a 2 line "
                      "context of the file. Keep your response in 2 lines. If text is in other language "
                      f"than English mention that and keep the context summary in English only:\n{full_text}")
    if len(summary_prompt) > 32000:
        summary_prompt = summary_prompt[:32000]
    sys_msg = SystemMessage(summary_prompt)
    summary = model.invoke([sys_msg]).content

    embeddings = AzureOpenAIEmbeddings(
        model=AZURE_EMBEDDING_MODEL,
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_key=AZURE_EMBEDDING_KEY
    )
    if delete_chunks:
        graph.query("""MATCH (n:Chunk{origin_filename: $file_name, username: $username}) DETACH DELETE n""",
                    params={"file_name": file_name, "username": username})
    if image_data is None and file_abs_path is not None:
        # Read the image file in binary mode
        with open(file_abs_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Generate Chunk ids
    id_list, i = [], 0
    for doc in split_documents:
        key = str(doc.metadata['origin_filename']) + str(doc.metadata['username']) + str(i)
        id_list.append(hashlib.md5(key.encode('utf-8')).hexdigest())
        i += 1

    db = Neo4jVector.from_documents(
        split_documents,
        embeddings,
        ids=id_list
    )

    if image_data is not None:
        graph.query("""MERGE (f:File {name: $file_name, username: $username})
            ON CREATE SET f.timestamp = $timestamp, f.date = $date, f.type = $type, f.summary = $summary, f.data = $data, f.username = $username
            ON MATCH SET f.timestamp = $timestamp, f.date = $date, f.type = $type, f.summary = $summary, f.data = $data, f.username = $username
            WITH f    
            MATCH (c:Chunk {origin_filename: $file_name, username: $username})
            MERGE (f)-[:CHUNKED_INTO]->(c)""",
                    params={
                        "file_name": file_name,
                        "timestamp": current_timestamp,
                        "type": split_documents[0].metadata['format'],
                        "date": current_timestamp[:10],
                        "summary": summary,
                        "data": image_data,
                        "username": username
                    }
                    )
    else:
        graph.query(
            """MERGE (f:File {name: $file_name, username: $username})
            ON CREATE SET f.timestamp = $timestamp, f.date = $date, f.type = $type, f.summary = $summary, f.username = $username
            ON MATCH SET f.timestamp = $timestamp, f.date = $date, f.type = $type, f.summary = $summary, f.username = $username
            WITH f    
            MATCH (c:Chunk {origin_filename: $file_name, username: $username})
            MERGE (f)-[:CHUNKED_INTO]->(c)""",
            params={
                "file_name": file_name,
                "timestamp": current_timestamp,
                "type": split_documents[0].metadata['format'],
                "date": current_timestamp[:10],
                "summary": summary,
                "username": username
            }
        )

    # Link to current user
    graph.query("""MATCH (f:File {name: $file_name, username: $username})
        MATCH (u:User {username: $username})
        WITH f, u
        MERGE (u)-[:UPLOADED_FILE]->(f)""",
                params={
                    "file_name": file_name,
                    "username": username
                }
                )
    return {"name": file_name, "type": split_documents[0].metadata['format'], "summary": summary}


def process_chart(chart_detail, image_name, image_bytes, i, page_num, img_index, file, username):
    chart_data = img_ops.get_details_from_chart(image_name, chart_detail, image_bytes)
    chart_detail['data'] = chart_data
    doc = Document(page_content=str(chart_detail))
    doc.metadata['chunk_no'] = i
    if page_num is not None:
        doc.metadata['image_id'] = str(page_num) + '.' + str(img_index) + '.' + str(i)
    doc.metadata['chunk_create_ts'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    doc.metadata['origin_filename'] = file.split('\\')[-1] if '\\' in file else file.split('/')[-1]
    doc.metadata['format'] = file.split('.')[-1]
    doc.metadata['username'] = username
    try:
        for key, val in chart_detail.items():
            doc.metadata[key] = val
    except:
        pass
    doc.metadata = {k: v for k, v in doc.metadata.items() if v != ''}

    return doc


def process_file(file, username):
    if file.endswith('.pdf'):
        loader = PyMuPDFLoader(file)

        # Image search
        pdf_doc = fitz.open(file)
        i = 1
        image_summary = ''
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base64.b64encode(base_image["image"]).decode('utf-8')
                error_message = ''
                image_summary_text = ''
                chart_summary_list = []
                image_name = file + '_image' + str(img_index)

                for retry in range(3):
                    try:
                        image_summary_text = img_ops.get_image_summary(image_name, image_bytes, error_message)
                        chart_summary_list = json.loads(image_summary_text)
                        break

                    except Exception as e:
                        error_message = str(traceback.print_exc(limit=1))

                if len(chart_summary_list) > 0:
                    split_documents = []
                    ip_params = []
                    for chart_detail in chart_summary_list:
                        ip_params.append(
                            (chart_detail, image_name, image_bytes, i, page_num, img_index, file, username))
                        i += 1

                    with ThreadPoolExecutor() as executor:
                        result = executor.map(lambda p: process_chart(*p), ip_params)
                    for doc in result:
                        split_documents.append(doc)

                    # Create Neo4j Nodes and Relations
                    summary_dict = create_file_and_chunks(file, username, image_summary_text, split_documents,
                                                          image_data=image_bytes)

                    if 'summary' in summary_dict:
                        image_summary += summary_dict['summary']
        pdf_doc.close()

        documents = loader.load()
        full_text = documents[0].page_content

        # Split the text into smaller chunks using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        split_documents = text_splitter.split_documents(documents)

        for doc in split_documents:
            doc.metadata['chunk_no'] = i
            doc.metadata['chunk_create_ts'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            doc.metadata['origin_filename'] = file.split('\\')[-1] if '\\' in file else file.split('/')[-1]
            doc.metadata['username'] = username
            doc.metadata = {k: v for k, v in doc.metadata.items() if v != ''}
            i += 1

        # Create Neo4j Nodes and Relations
        summary_dict = create_file_and_chunks(file, username, full_text, split_documents, delete_chunks=False)
        if image_summary != '':
            summary_dict['summary'] += 'File contains image, summary of those: ' + image_summary

        return summary_dict

    elif file.endswith('.doc') or file.endswith('.docx'):
        loader = UnstructuredWordDocumentLoader(file)
        documents = loader.load()
        full_text = documents[0].page_content

        # Split the text into smaller chunks using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        split_documents = text_splitter.split_documents(documents)

        i = 1
        for doc in split_documents:
            doc.metadata['chunk_no'] = i
            doc.metadata['chunk_create_ts'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            doc.metadata['origin_filename'] = file.split('\\')[-1] if '\\' in file else file.split('/')[-1]
            doc.metadata['format'] = file.split('.')[-1]
            doc.metadata['username'] = username
            doc.metadata = {k: v for k, v in doc.metadata.items() if v != ''}
            i += 1

        # Create Neo4j Nodes and Relations
        summary_dict = create_file_and_chunks(file, username, full_text, split_documents)

        return summary_dict
    elif file.endswith('.xlsx'):
        loader = UnstructuredExcelLoader(file)
        documents = loader.load()
        full_text = documents[0].page_content

        # Split the text into smaller chunks using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        split_documents = text_splitter.split_documents(documents)

        i = 1
        for doc in split_documents:
            doc.metadata['chunk_no'] = i
            doc.metadata['chunk_create_ts'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            doc.metadata['origin_filename'] = file.split('\\')[-1] if '\\' in file else file.split('/')[-1]
            doc.metadata['format'] = file.split('.')[-1]
            doc.metadata['username'] = username
            doc.metadata = {k: v for k, v in doc.metadata.items() if v != ''}
            i += 1

        # Create Neo4j Nodes and Relations
        summary_dict = create_file_and_chunks(file, username, full_text, split_documents)

        return summary_dict
    elif file.endswith('.png') or file.endswith('.jpeg'):
        error_message = ''
        image_summary_text = ''
        chart_summary_list = []

        for retry in range(3):
            try:
                image_summary_text = img_ops.get_image_summary(file, error_message=error_message)
                chart_summary_list = json.loads(image_summary_text)
                print(chart_summary_list)
                break

            except Exception as e:
                error_message = str(traceback.print_exc(limit=1))

        if len(chart_summary_list) > 0:
            i = 1
            split_documents = []
            ip_params = []
            for chart_detail in chart_summary_list:
                ip_params.append((chart_detail, file, None, i, None, None, file, username))
                i += 1

            with ThreadPoolExecutor() as executor:
                result = executor.map(lambda p: process_chart(*p), ip_params)
            for doc in result:
                split_documents.append(doc)

            # Create Neo4j Nodes and Relations
            summary_dict = create_file_and_chunks(file, username, image_summary_text, split_documents, file)

            return summary_dict


def process_given_files(files, username):
    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, files, [username] * len(files))

    summary_list = []
    for result in results:
        if result is not None:
            summary_list.append(result)

    return summary_list
