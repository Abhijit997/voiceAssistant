from langchain_openai import AzureChatOpenAI
import streamlit as st

AZURE_OPENAI_MODEL = st.secrets['AZURE_OPENAI_MODEL']
AZURE_OPENAI_ENDPOINT = st.secrets['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_KEY = st.secrets['AZURE_OPENAI_KEY']
AZURE_OPENAI_VERSION = st.secrets['AZURE_OPENAI_VERSION']

model = AzureChatOpenAI(
    model=AZURE_OPENAI_MODEL,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION,
)


def detect_emotion(text: str, user_query: str = '') -> str:
    prompt = (
        "Classify the emotional tone of the following sentence into one of the following styles: "
        "cheerful, sad, angry, excited, friendly, empathetic, hopeful, unfriendly, shouting, whispering, assistant, newscast, customerservice, narration-professional, narration-relaxed. Respond only with the style name.\n\n"
        f"User question: {user_query}"
        f"Received Response that need to be dictated: \"{text}\""
    )

    try:
        response = model.invoke([
            {"role": "system",
             "content": "Only respond with one word: cheerful, sad, angry, excited, friendly, empathetic, hopeful, unfriendly, shouting, whispering, assistant, newscast, customerservice, narration-professional, narration-relaxed"},
            {"role": "user", "content": prompt}
        ])
        style = response.content.strip().lower()
        allowed_styles = ["cheerful", "sad", "angry", "excited", "friendly", "empathetic", "hopeful", "unfriendly",
                          "shouting", "whispering", "assistant", "newscast", "customerservice",
                          "narration-professional", "narration-relaxed"]
        return style if style in allowed_styles else "friendly"
    except Exception as e:
        print(f"[Emotion Detection Failed] {e}")
        return "friendly"


def build_ssml(text, voice="en-US-JennyNeural", rate="medium", style="friendly", pitch="0%", volume="0dB"):
    if style != "friendly":
        return f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
               xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice}">
                <mstts:express-as style="{style}">
                    <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
                        {text}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>
        """
    else:
        return f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice}">
                <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
