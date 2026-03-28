import base64
import streamlit as st
from langchain_openai import ChatOpenAI
from ai_agent.utils import is_file_size_valid, MAX_IMAGE_SIZE_MB

try:
    from dotenv import load_dotenv

    load_dotenv()
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
        if not is_file_size_valid(upload_file.size):
            st.error(f"The uploaded image exceeds the {MAX_IMAGE_SIZE_MB}MB limit.")
            return

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
            st.text(user_input)
            st.image(upload_file)
            st.markdown("### Answer")
            st.write_stream(llm.stream(query))
    else:
        st.write("Please upload an image file.")


if __name__ == "__main__":
    main()
