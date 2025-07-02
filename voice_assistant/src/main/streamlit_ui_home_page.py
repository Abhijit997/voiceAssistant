import asyncio
import base64
import datetime
import hashlib
import json
import os
import tempfile
import uuid
from io import BytesIO

import azure.cognitiveservices.speech as speechsdk
import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage

from operations_file_chunk_node import process_given_files
from operations_langgraph import build_graph
from operations_user_chat_node import save_chat, load_last_3_chats
from operations_voices import build_ssml, detect_emotion

# Azure Speech Credentials
AZURE_SPEECH_KEY = st.secrets['AZURE_SPEECH_KEY']
AZURE_REGION = st.secrets['AZURE_REGION']

# Define allowed file types
ALLOWED_FILE_TYPES = [
    "text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "image/png", "image/jpeg"
]
OUTPUT_LANGUAGE_LIST = [
    "English (India)", "English (US)", "Hindi", "Tamil", "Telugu", "Marathi", "French", "Spanish", "Arabic",
    "German", "Chinese (Mandarin)", "Japanese", "Korean", "Russian", "Portuguese (Brazil)", "Italian"
]

LANGUAGE_OPTIONS = {
    "English (India)": {
        "code": "en-IN",
        "voice": "en-IN-NeerjaNeural"
    },
    "English (US)": {
        "code": "en-US",
        "voice": "en-US-JennyNeural"
    },
    "Hindi": {
        "code": "hi-IN",
        "voice": "hi-IN-SwaraNeural"
    },
    "Tamil": {
        "code": "ta-IN",
        "voice": "ta-IN-PallaviNeural"
    },
    "Telugu": {
        "code": "te-IN",
        "voice": "te-IN-MohanNeural"
    },
    "Marathi": {
        "code": "mr-IN",
        "voice": "mr-IN-AarohiNeural"
    },
    "Bengali": {
        "code": "bn-IN",
        "voice": "bn-IN-PradeepNeural"
    },
    "French": {
        "code": "fr-FR",
        "voice": "fr-FR-DeniseNeural"
    },
    "Spanish": {
        "code": "es-ES",
        "voice": "es-ES-ElviraNeural"
    },
    "Arabic": {
        "code": "ar-EG",
        "voice": "ar-EG-SalmaNeural"
    },
    "German": {
        "code": "de-DE",
        "voice": "de-DE-KatjaNeural"
    },
    "Chinese (Mandarin)": {
        "code": "zh-CN",
        "voice": "zh-CN-XiaoxiaoNeural"
    },
    "Japanese": {
        "code": "ja-JP",
        "voice": "ja-JP-NanamiNeural"
    },
    "Korean": {
        "code": "ko-KR",
        "voice": "ko-KR-SoonBokNeural"
    },
    "Russian": {
        "code": "ru-RU",
        "voice": "ru-RU-SvetlanaNeural"
    },
    "Portuguese (Brazil)": {
        "code": "pt-BR",
        "voice": "pt-BR-FranciscaNeural"
    },
    "Italian": {
        "code": "it-IT",
        "voice": "it-IT-ElsaNeural"
    }
}


def speak(text, user_query, voice='en-US-JennyNeural'):
    # Azure Speech Config
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)

    # Synthesize speech to audio stream
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    # result = synthesizer.speak_text_async(text).get()

    # voice_choice = st.session_state['voice_name']  # or "en-US-JennyNeural" etc.
    # voice_choice = "en-US-JennyNeural"

    style = detect_emotion(text, user_query)
    print(style)
    # ssml = build_ssml(text, voice=voice, rate="medium" , style=style)
    ssml = build_ssml(
        text,
        voice=voice,
        rate=st.session_state["ssml_rate"],
        pitch=st.session_state["ssml_pitch"],
        volume=st.session_state["ssml_volume"],
        style=style
    )
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Save audio stream to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
            audio_file.write(result.audio_data)
            audio_path = audio_file.name

        # Read and base64 encode the audio
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            b64 = base64.b64encode(audio_bytes).decode()

        # Generate unique ID for audio tag
        unique_id = str(uuid.uuid4()).replace('-', '')

        # Embed and autoplay in Streamlit
        st.markdown(
            f"""
            <audio id="{unique_id}" autoplay>
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            <script>
                var audioElem = document.getElementById("{unique_id}");
                if (audioElem) {{
                    audioElem.load();
                    audioElem.play();
                }}
            </script>
            """,
            unsafe_allow_html=True
        )

        # Clean up temp file
        os.remove(audio_path)

    else:
        st.error("Failed to synthesize speech.")


# Define a callback function to switch labels
def switch_label():
    if st.session_state["button_state"]:
        st.session_state["button_label"] = "🔇"
    else:
        st.session_state["button_label"] = "🔊"
    st.session_state["button_state"] = not st.session_state["button_state"]


# React to user input
def generate_response(react_graph, chat_container, prompt, output_lang, output_voice):
    with chat_container:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        current_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z %A')
        current_chat_id = hashlib.sha256(
            (st.session_state['logged_user_details']['first_name'] + current_timestamp)
            .encode('utf-8')).hexdigest()

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Stream assistants response
        if st.session_state["button_state"]:
            messages = [
                HumanMessage(
                    content=f"timestamp: {current_timestamp}, user_placed_files: {st.session_state['processed_files']}, "
                            f"output_format: user wants response to be short and easy to read out loud as audio, do not include any | or - for tables or any text formatting"
                            f"output_language: {output_lang}, message: {prompt}",
                    metadata={"timestamp": current_timestamp, "user_placed_files": st.session_state['processed_files'],
                              "output_language": output_lang, "user_details": st.session_state['logged_user_details']}
                )
            ]
        else:
            messages = [
                HumanMessage(
                    content=f"timestamp: {current_timestamp}, user_placed_files: {st.session_state['processed_files']}, "
                            f"output_format: user wants response in more detail and not in audio format, put bullet points or tables whenever possible, "
                            f"output_language: {output_lang}, message: {prompt}",
                    metadata={"timestamp": current_timestamp, "user_placed_files": st.session_state['processed_files'],
                              "output_language": output_lang, "user_details": st.session_state['logged_user_details']}
                )
            ]
        messages[0].pretty_print()
        config = {"configurable": {"thread_id": "1"}}

        with st.spinner("Waiting for response...", show_time=True):
            for response in react_graph.stream({"messages": messages}, config=config):
                if 'tools' in response:
                    response['tools']['messages'][0].pretty_print()
                    try:
                        tool_response = json.loads(response['tools']['messages'][0].content)
                        if 'metadata' in tool_response and 'image_data' in tool_response['metadata'] and isinstance(
                                tool_response['metadata']['image_data'], dict):
                            for name, image_utf in tool_response['metadata']['image_data'].items():
                                image_bytes = BytesIO(base64.b64decode(image_utf))
                                image = Image.open(image_bytes)
                                st.image(image, caption=name)
                                st.session_state.messages.append(
                                    {"role": "image", "content": image_bytes, "name": name})
                    except Exception as ee:
                        print(ee)
                # Response coming from assistant
                elif 'assistant' in response:
                    content = response['assistant']['messages'][0].content
                    response['assistant']['messages'][0].pretty_print()
                    # Thinking message generation
                    if isinstance(content, list):
                        pass
                    # Tool call message
                    elif response['assistant']['messages'][0].tool_calls:
                        pass
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(content.replace("$", "\\$"))
                            st.session_state.messages.append(
                                {"role": "assistant", "content": content.replace("$", "\\$")})

                            # Store query-response in db
                            chat = {'user_first_name': st.session_state['logged_user_details']['first_name'],
                                    'username': st.session_state['logged_user_details']['username'],
                                    'user_timezone': datetime.datetime.now().astimezone().tzname(),
                                    'user_query': prompt,
                                    'agent_response': content,
                                    'id': current_chat_id,
                                    'timestamp': current_timestamp}
                            asyncio.run(save_chat(chat, st.session_state['logged_user_details']['username'],
                                                  st.session_state['last_chat_id']))
                            st.session_state['last_chat_id'] = current_chat_id

                            # Speech output when st.session_state["button_state"] == True
                            if st.session_state["button_state"]:
                                speak(content, prompt, output_voice)

                else:
                    pass
                    # print(response)


def home_page():
    st.title(f"Welcome {st.session_state['logged_user_details']['first_name']}")

    # Azure Speech Config
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)

    # Create 2 columns for chat box and microphone button
    with st._bottom:
        col1, col2, col3 = st.columns([10, 1, 1])
    # Define the main container for the chat
    chat_container = st.container()
    # Add custom CSS to position the columns at the bottom

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize the file summary variable if it doesn't exist
    if 'processed_files' not in st.session_state:
        st.session_state['processed_files'] = []

    if 'ssml_rate' not in st.session_state:
        st.session_state["ssml_rate"] = 'medium'
    if 'ssml_pitch' not in st.session_state:
        st.session_state["ssml_pitch"] = 'default'
    if 'ssml_volume' not in st.session_state:
        st.session_state["ssml_volume"] = 'default'

    if 'last_3_chat_contents' not in st.session_state:
        st.session_state['last_3_chat_contents'] = load_last_3_chats(
            st.session_state['logged_user_details']['username'])
    elif st.session_state['last_3_chat_contents'] is None:
        st.session_state['last_3_chat_contents'] = load_last_3_chats(
            st.session_state['logged_user_details']['username'])

    # Display sidebar for file upload
    st.sidebar.title("Upload Files for Insights")
    st.sidebar.badge("ℹ️ Supported file types: PDF, Word Doc, Excel Doc, JPEG, PNG")

    with chat_container:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            if message['content'] is None or message['content'] == '':
                pass
            elif message['role'] == 'image':
                st.image(Image.open(message["content"]), caption=message["name"])
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Initialize session state variables
    if "button_label" not in st.session_state:
        st.session_state["button_label"] = "🔇"
        st.session_state["button_state"] = False

    # Add file upload button
    uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

    # File uploader
    processed_file_set = set(y['name'] for y in st.session_state['processed_files'])
    if len(uploaded_files) > 0:
        if not st.session_state['processed_files']:
            new_files = uploaded_files
        elif 'name' not in st.session_state['processed_files'][0]:
            new_files = uploaded_files
        else:
            new_files = [x for x in uploaded_files if x.name not in processed_file_set]

        new_files_path = []
        for new_file in new_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.'+new_file.name.split('.')[-1]) as audio_file:
                audio_file.write(new_file.getbuffer())
                new_files_path.append(audio_file.name)

        st.write(new_files_path)

        if len(new_files_path) > 0:
            response = process_given_files(new_files_path, st.session_state['logged_user_details']['username'])
            st.write(response)
            if (isinstance(response, list)) and len(response) > 0:
                if response[0] is None:
                    st.error('Files processing failed!')
                else:
                    st.success(', '.join(str(x['name']) for x in response) + ' processed successfully!')
            else:
                st.error('Files processing failed!')

            # Update st.session_state['processed_files'] from response
            if response is None:
                pass
            elif len(st.session_state['processed_files']) == 0:
                st.session_state['processed_files'] = response
            else:
                for user_file in response:
                    if user_file['name'] not in processed_file_set:
                        st.session_state['processed_files'].extend(user_file)

    input_lang = st.sidebar.selectbox("🎤 Select Input Language (speech recognition)", list(LANGUAGE_OPTIONS.keys()),
                                      index=0)
    output_lang = st.sidebar.selectbox("🗣️ Select Output Language (spoken reply)", OUTPUT_LANGUAGE_LIST, index=0)

    st.sidebar.markdown("🎚️ **Voice Controls**")

    # 🌀 SSML Rate (Speed)
    rate_slider_val = st.sidebar.select_slider(label="🌀 Speed (Rate)",
                                               options=['0.25x', '0.5x', '1x', '1.5x', '2x'],
                                               value='1x')
    rate_map = {'0.25x': "x-slow", '0.5x': "slow", '1x': "medium", '1.5x': "fast", '2x': "x-fast"}
    ssml_rate = rate_map[rate_slider_val]

    # 🎵 SSML Pitch
    pitch_slider_val = st.sidebar.slider("🎵 Pitch", -2, 2, 0)
    pitch_map = {-2: "x-low", -1: "low", 0: "default", 1: "medium", 2: "high", 3: "x-high"}
    ssml_pitch = pitch_map[pitch_slider_val]

    # 🔊 SSML Volume
    volume_slider_val = st.sidebar.slider("🔊 Volume", 0, 6, 3)
    volume_map = {
        0: "silent",
        1: "x-soft",
        2: "soft",
        3: "default",
        4: "medium",
        5: "loud",
        6: "x-loud",
    }
    ssml_volume = volume_map[volume_slider_val]

    # Store in session state
    st.session_state["ssml_rate"] = ssml_rate
    st.session_state["ssml_pitch"] = ssml_pitch
    st.session_state["ssml_volume"] = ssml_volume

    # Last 3 chats
    st.sidebar.write('Conversation History (Recent 3):')
    for conversation in st.session_state['last_3_chat_contents']:
        with st.sidebar.expander(conversation['timestamp'][:19] + ' - '
                                 + (conversation['chat_content'][0]['user_query']
        if len(conversation['chat_content'][0]['user_query']) < 25
        else conversation['chat_content'][0]['user_query'][:25] + '...')):
            for chat in conversation['chat_content']:
                pass
                st.markdown(':red[' + st.session_state['logged_user_details']['first_name']
                            + ']: ' + chat['user_query'])
                st.markdown(':green[Assistant]: ' + chat['agent_response'].replace("$", "\\$"))
                st.divider()

    side_col1, side_col2 = st.sidebar.columns([2, 1])

    # Start new session button
    with side_col1:
        if st.button("Start New Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state['last_chat_id'] = None
            st.session_state['last_3_chat_contents'] = None
            st.session_state['react_graph'] = build_graph(st.session_state['logged_user_details'])
            st.rerun()

    # Logout button
    with side_col2:
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state.messages = []
            st.session_state['processed_files'] = []
            st.session_state['logged_user_details'] = {}
            st.session_state['last_chat_id'] = None
            st.session_state['last_3_chat_contents'] = None
            st.rerun()

    input_lang_code = LANGUAGE_OPTIONS[input_lang]["code"]
    output_voice = LANGUAGE_OPTIONS[output_lang]["voice"]

    with col1:
        if prompt := st.chat_input("Ask your questions here"):
            generate_response(st.session_state['react_graph'], chat_container, prompt, output_lang, output_voice)

    with col2:
        if st.button("🎤", use_container_width=True):
            # Speech to Text
            audio_config = speechsdk.AudioConfig(use_default_microphone=True)
            speech_config.speech_recognition_language = input_lang_code
            with chat_container:
                st.info('Listening...')
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                prompt = result.text
                generate_response(st.session_state['react_graph'], chat_container, prompt, output_lang, output_voice)
            else:
                with chat_container:
                    st.error("Speech not recognized")
                    # time.sleep(2)
            # st.rerun()

    with col3:
        if st.button(st.session_state["button_label"], on_click=switch_label, use_container_width=True):
            pass

# if __name__ == '__main__':
#    main()
