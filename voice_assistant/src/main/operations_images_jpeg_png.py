import base64
import streamlit as st
from openai import AzureOpenAI

# === CONFIGURATION ===
# Computer Vision Config


AZURE_OPENAI_MODEL = st.secrets['AZURE_OPENAI_MODEL']
AZURE_OPENAI_ENDPOINT = st.secrets['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_KEY = st.secrets['AZURE_OPENAI_KEY']
AZURE_OPENAI_VERSION = st.secrets['AZURE_OPENAI_VERSION']

# Initialize the OpenAI LLM with Azure configuration
model = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION
)


def get_image_summary(image_path, image_data=None, error_message=None):
    prompt = (
        "You are a helpful assistant specialized on images that contain charts. You are to find presence of these "
        "17 types of charts on given image: area_map, combo, donut, gauge, multi_row_card, ribbon, scatter, table, "
        "bar_chart, card_chart, column_chart, funnel, line_chart, map, pie_chart, treemap, waterfall, other. "
        "For each type of chart you find add a new element on your output JSON list. Each element will have 4 keys: "
        'title, type, description, relative_position. In description attribute include all the minimum to maximum limits '
        'you see for each axis. Output only the JSON message nothing else. Example output: '
        '[{"title": "This Year Sales and Last Year Sales by FiscalMonth", '
        '"type":"area_map", "description":"This is a line and area chart comparing sales for \"This Year\" (in blue) '
        'and \"Last Year\" (in red) by fiscal month from January to August. The X-axis represents months from Jan '
        'till Aug. The Y-axis represents sales in millions of dollars (from $2M to $4M)", "relative_position":'
        '"top left"}, {"title":"Total Sales Variance % and Sales Per Sq Ft by District", "type":"scatter", '
        '"description":"The image is a scatter plot graph titled **\"Total Sales Variance % and Sales Per Sq Ft by '
        'District\"**. The y-axis represents **Sales Per Sq Ft** (ranges from $12 to $15). The x-axis represents '
        '**Total Sales Variance %** (ranges from -10% to 0%). Each point on the graph represents a district (e.g., '
        'FD - 01, LI - 01) and is positioned based on its sales variance and sales per sq ft value.", '
        '"relative_position":"top right"}]')
    if error_message is not None and len(error_message) > 0:
        prompt += f'\nOn last try you had following error {error_message}, please correct it'

    if image_data is None:
        # Read the image file in binary mode
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

    response = model.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_path.split('.')[-1]};base64,{image_data}"
                        }
                    },
                ],
            }
        ]
    )

    return response.choices[0].message.content


def get_details_from_chart(image_path, image_detail, image_data=None, error_message=None):
    prompt = (
        "You are a helpful assistant specialized on extracting data from images in comma separated format. You are "
        "given an image that contains charts in it. According to following instructions fetch the corresponding "
        f"chart details from the image and output it as CSV, example: "
        "Month,This Year Sales ($M),Last Year Sales ($M)\n"
        "Jan,1.5,2.0\nFeb,2.5,2.5\nMar,3.8,2.8\nApr,2.7,3.3\nMay,2.7,2.7\nJun,3.1,3.0\nJul,2.2,3.3\nAug,3.3,3.6\n"
        f"Chart details: {str(image_detail)}")

    if error_message is not None and len(error_message) > 0:
        prompt += f'\nOn last try you had following error {error_message}, please correct it'

    if image_data is None:
        # Read the image file in binary mode
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

    response = model.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_path.split('.')[-1]};base64,{image_data}"
                        }
                    },
                ],
            }
        ]
    )

    return response.choices[0].message.content
