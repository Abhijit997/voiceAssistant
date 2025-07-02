import os

STATIC_DIR = 'static'
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

import streamlit as st
import streamlit_ui_home_page
import streamlit_ui_login

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ip_address = s.getsockname()[0]
print(ip_address)

NEO4J_URI = st.secrets['NEO4J_URI']
NEO4J_USER = st.secrets['NEO4J_USER']
NEO4J_PASSWORD = st.secrets['NEO4J_PASSWORD']
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if 'logged_user_details' not in st.session_state:
        st.session_state['logged_user_details'] = {}

    if 'last_chat_id' not in st.session_state:
        st.session_state['last_chat_id'] = None

    if 'react_graph' not in st.session_state:
        st.session_state['react_graph'] = None

    if not st.session_state['logged_in']:
        streamlit_ui_login.login_page()
    else:
        if st.session_state['logged_user_details'] == {}:
            st.error('User Details not provided, cannot proceed to homepage.')
        else:
            streamlit_ui_home_page.home_page()


if __name__ == '__main__':
    main()
