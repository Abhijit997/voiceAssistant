import hashlib
import streamlit as st
from langchain_neo4j import Neo4jGraph
from operations_langgraph import build_graph


# Function to handle login
def login(username, password):
    graph = Neo4jGraph()
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()

    # Match for username password
    login_validation_response = graph.query('''MATCH (n:User{username:$username, password:$password}) RETURN n''',
                                            params={'username': username, 'password': hashed_password})

    if len(login_validation_response) > 0:
        st.session_state['logged_user_details'] = login_validation_response[0]['n']
        del st.session_state['logged_user_details']['password']
        return True
    else:
        return False


# Function to handle registration
def register(first_name, last_name, role, username, password):
    graph = Neo4jGraph()

    # Check user already exists
    check_existing_response = graph.query('''MATCH (n:User{username: $username}) RETURN COUNT(n) AS COUNT''',
                                          params={'username': username})
    if check_existing_response[0]['COUNT'] > 0:
        st.error("User already exists")

    # Create User node in neo4j
    else:
        hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        user_creation_response = graph.query('''CREATE (:User {
        first_name: $first_name, last_name: $last_name, role: $role, 
        username: $username, password: $password
        })''', params={'first_name': first_name, 'last_name': last_name, 'role': role,
                       'username': username, 'password': hashed_password})

        if len(user_creation_response) == 0:
            st.success('User created successfully, please proceed to Login')
        else:
            st.error('User creation failed')


# Login page function to render the login page
def login_page():
    st.title("Voice Assistant")
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # if 'login_type' not in st.session_state:
    #    st.session_state['login_type'] = 'Register'
    if 'login_flag' not in st.session_state:
        st.session_state['login_flag'] = True

    col1, col2, col3 = st.columns([6.1, 1.2, 2])
    with col2:
        if st.session_state['login_flag'] == True:
            register_button = st.button('Register', key="register_button", use_container_width=True)
        else:
            register_button = st.button('Login', key="register_button", use_container_width=True)
    with col3:
        forget_password_button = st.button("Forgot Password", key="forget_password_button", use_container_width=True)

    if register_button:
        st.session_state['login_flag'] = not st.session_state['login_flag']
        st.rerun()

    # Handle register form
    if not st.session_state['login_flag']:
        st.subheader("New User Registration")
        col1, col2, col3 = st.columns([1, 1, 1])
        new_first_name, new_last_name, new_role = '', '', 'User'
        with col1:
            new_first_name = st.text_input("First Name", key="first_name")
        with col2:
            new_last_name = st.text_input("Last Name", key="last_name")
        with col3:
            new_role = st.selectbox('User Role', ['Admin', 'User'], index=1)

        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type='password', key="register_password")
        if st.button("Register", key="register_submit"):
            register(new_first_name, new_last_name, new_role, new_username, new_password)
    # Handle login form
    else:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['react_graph'] = build_graph(st.session_state['logged_user_details'])
                st.rerun()
            elif password == '':
                st.error("Password is empty")
            else:
                st.error("Invalid Username/Password")
