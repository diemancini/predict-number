import base64
import json
import os
import uuid
from io import BytesIO
from pathlib import Path
import pandas as pd
import requests

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

API_BASE_URL = os.getenv("WEB_BASE_URL")
API_READ_LOGS_URL = "predict-number/read/logs"
API_READ_NUMBER_PREDICTION_URL = "predict-number/read/number"

IS_LOADED_PAGE = "is_loaded_page"
ST_CANVAS = "st_canvas"
CURRENT_PREDICTION = "current_prediction"
PREDICTION_KEY = "prediction"
CONFIDENCE_KEY = "confidence"
LABEL_KEY = "label"


def main():
    if CURRENT_PREDICTION not in st.session_state:
        st.session_state[CURRENT_PREDICTION] = {
            PREDICTION_KEY: None,
            CONFIDENCE_KEY: None,
            LABEL_KEY: None,
        }

    col_draw, col_container = st.columns(2)

    b64_encoded = ""
    with col_draw:
        b64_encoded = png_export()
    with col_container.container(height=400, border=False):
        if None not in st.session_state[CURRENT_PREDICTION].values():
            st.write(
                f"Prediction: {st.session_state[CURRENT_PREDICTION][PREDICTION_KEY]}"
            )
            st.write(
                f"Confidence: {float(st.session_state[CURRENT_PREDICTION][CONFIDENCE_KEY]):.2f}%"
            )
        else:
            st.write(f"Prediction:")
            st.write(f"Confidence:")

        col_label, col_text_input = st.columns([4, 1])
        col_label.write("True Label")
        true_label = col_text_input.text_input(
            label="True Label", value="", label_visibility="collapsed"
        )

        st.button(
            "Submit", on_click=read_number_prediction, args=[b64_encoded, true_label]
        )

    # Get logs from api and print in page!
    logs_response = get_prediction_logs()
    if logs_response:
        df = pd.DataFrame(logs_response)
        del df[CONFIDENCE_KEY]
        st.table(df)


@st.dialog("ERROR")
def submit_error():
    st.write(f"Please insert a number between 0-9 on TRUE LABEL field!")
    if st.button("Ok"):
        st.rerun()


def png_export():
    try:
        Path("tmp/").mkdir()
    except FileExistsError:
        pass

    file_path = f"tmp/{uuid.uuid4()}.png"
    data = st_canvas(
        update_streamlit=True,
        key="image_export",
        background_color="rgba(134,143,152,255)",
        height=300,
        width=300,
        stroke_color="white",
        stroke_width=20,
    )
    b64 = ""
    if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        im.save(file_path, "PNG")

        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_data = buffered.getvalue()

        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(img_data).decode()
        except AttributeError as e:
            print(e)

        os.remove(file_path)

    return b64


def get_prediction_logs():
    url = f"{API_BASE_URL}/{API_READ_LOGS_URL}"
    # st.write(url)
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        error_details = str(response.json()["detail"]).upper()
        st.error(error_details, icon="🚨")
        return None
    return response.json()


def read_number_prediction(b64_encoded, true_label):

    if not true_label:
        # st.error("This is an error", icon="🚨")
        submit_error()
        return None

    url = f"{API_BASE_URL}/{API_READ_NUMBER_PREDICTION_URL}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "label": true_label,
        "b64_encoded": b64_encoded,
    }
    # Send the POST request with headers
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        response_obj = response.json()
        st.session_state[CURRENT_PREDICTION] = {
            PREDICTION_KEY: response_obj[PREDICTION_KEY],
            CONFIDENCE_KEY: response_obj[CONFIDENCE_KEY],
            LABEL_KEY: response_obj[PREDICTION_KEY],
        }
    # TODO: COMPLETE THE EXCEPTIONS FOR RESPONSE


if __name__ == "__main__":
    st.set_page_config(page_title="Digit Recognizer", page_icon=":pencil2:")
    st.title("Digit Recognizer")
    main()
