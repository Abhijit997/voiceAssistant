import os

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings
import streamlit as st

NEO4J_URI = st.secrets['NEO4J_URI']
NEO4J_USER = st.secrets['NEO4J_USER']
NEO4J_PASSWORD = st.secrets['NEO4J_PASSWORD']
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

AZURE_EMBEDDING_MODEL = st.secrets['AZURE_EMBEDDING_MODEL']
AZURE_EMBEDDING_ENDPOINT = st.secrets['AZURE_EMBEDDING_ENDPOINT']
AZURE_EMBEDDING_KEY = st.secrets['AZURE_EMBEDDING_KEY']

async def save_chat(chat_dict, username, prev_chat_id=None):
    graph = Neo4jGraph()
    embeddings = AzureOpenAIEmbeddings(
        model=AZURE_EMBEDDING_MODEL,
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_key=AZURE_EMBEDDING_KEY,
    )
    embedding_dict = {key: value for key, value in chat_dict.items() if key != 'id'}
    embedding_vector = embeddings.embed_query(str(embedding_dict))
    chat_dict['embedding'] = embedding_vector

    properties = ''
    for k, v in chat_dict.items():
        properties += k + ":'" + str(v).replace("'", "\\'") + "',"
    properties = properties[:-1]

    if prev_chat_id is None:
        cypher_message = ("CREATE (c:Chat {" + properties + "}) WITH c MATCH (u:User{username:'" + username
                          + "'}) WITH c, u CREATE (u)-[:CONVERSED]->(c)")
        graph.query(cypher_message)
    else:
        cypher_message = ("CREATE (c1:Chat {" + properties + "}) WITH c1 MATCH (c2:Chat{id:'" + prev_chat_id
                          + "'}) WITH c1, c2 CREATE (c2)-[:FOLLOWED_BY]->(c1)")
        graph.query(cypher_message)


def load_last_3_chats(username):
    graph = Neo4jGraph()

    cypher_message = ("MATCH (u:User{username:'" + username + "'})-[CONVERSED]->(c:Chat) "
                                                              "WITH u, c ORDER BY c.timestamp DESC LIMIT 3 "
                                                              "MATCH (c:Chat)-[FOLLOWED_BY*0..]->(d:Chat) "
                                                              "WITH c.timestamp AS timestamp, {user_query: d.user_query, agent_response: d.agent_response} AS combined "
                                                              "WITH timestamp, COLLECT(combined) AS chat_content "
                                                              "RETURN timestamp, chat_content")
    return graph.query(cypher_message)
