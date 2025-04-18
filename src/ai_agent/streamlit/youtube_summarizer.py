import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import YoutubeLoader
from urllib.parse import urlparse

SUMMARIZE_PROMPT = """
以下のコンテンツについて、内容を300文字程度で、できるだけわかりやすく要約してください。

==========
{content}
==========

回答は、日本語で行うこと。
"""


def init_page():
    st.set_page_config(
        page_title="Youtube▶️ Summarizer",
        page_icon="▶️",
        layout="wide",
    )
    st.title("Youtube▶️ Summarizer")
    st.sidebar.title("Options")


def select_model():
    """利用するLLMをサイドバーの選択状態によって切り替える"""
    temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1
    )
    models = (
        "Open AI GPT-3.5-turbo",
        "Open AI GPT-4o",
        "Claude 3.5 Haiku",
        "Google Gemini 1.5 Flash",
    )
    model = st.sidebar.radio("Choose a model:", models)

    match model:
        case "Open AI GPT-3.5-turbo":
            st.session_state.model_name = "gpt-3.5-turbo"
            return ChatOpenAI(
                model=st.session_state.model_name, temperature=temperature
            )
        case "Open AI GPT-4o":
            st.session_state.model_name = "gpt-4o"
            return ChatOpenAI(
                model=st.session_state.model_name, temperature=temperature
            )
        case "Claude 3.5 Haiku":
            st.session_state.model_name = "claude-3-5-haiku-20241022"
            return ChatAnthropic(
                model=st.session_state.model_name, temperature=temperature
            )
        case "Google Gemini 1.5 Flash":
            st.session_state.model_name = "gemini-1.5-flash-latest"
            return ChatGoogleGenerativeAI(
                model=st.session_state.model_name, temperature=temperature
            )
        case _:
            raise ValueError("Invalid model selected.")


def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", SUMMARIZE_PROMPT),
        ]
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain


def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    """
    Document:
        - page_content: str
        - metadata: dict
            - source: str
            - title: str
            - description: Optional[str]
            - view_count: int
            - thumbnail_url: Optional[str]
            - publish_date: str
            - length: int
            - author: str
    """
    with st.spinner("Fetching content..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=["en", "ja"],
        )
        res = loader.load()
        try:
            if res:
                content = res[0].page_content
                title = res[0].metadata["title"]
                return f"Title: {title}\n\n{content}"
            else:
                return None
        except Exception as e:
            st.error(f"Failed to fetch content from the URL: {e}")
            print(traceback.format_exc())
            return None


def main():
    init_page()
    chain = init_chain()

    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)

        if not is_valid_url:
            st.error("無効なURLです。")
            return

        if content := get_content(url):
            st.markdown("## Summary")
            st.write_stream(chain.stream({"content": content}))
            st.markdown("---")
            st.markdown("## Original Content")
            st.write(content)


if __name__ == "__main__":
    main()
