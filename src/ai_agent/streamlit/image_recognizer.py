import base64
import streamlit as st
from langchain_openai import ChatOpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
    # [debug]check envs
    # import os

    # environment_variables = os.environ
    # for key, value in environment_variables.items():
    #     print(f"{key}={value}")
except ImportError:
    import warnings

    warnings.warn(
        "dotenv not found. Please make sure to set your environment variables manually.",
        ImportWarning,
    )


def init_page():
    st.set_page_config(page_title="Image Recognizer")
    st.header("Image Recognizer")
    st.sidebar.title("Options")


def main():
    init_page()
    llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=512)
    upload_file = st.file_uploader(
        label="Upload an image",
        type=["jpg", "jpeg", "png"],
    )
    if upload_file:
        if user_input := st.text_input("Enter a description of the image"):
            image_base64 = base64.b64encode(upload_file.read()).decode()
            image = f"data:image/jpeg;base64,{image_base64}"
            query = [
                (
                    "user",
                    [
                        {"type": "text", "text": user_input},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                                "detail": "auto",
                            },
                        },
                    ],
                )
            ]
            st.markdown("### Question")
            st.write(user_input)
            st.image(upload_file)
            st.markdown("### Answer")
            st.write_stream(llm.stream(query))
    else:
        st.write("Please upload an image file.")


if __name__ == "__main__":
    main()
