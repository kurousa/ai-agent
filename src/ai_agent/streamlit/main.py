import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from dotenv import load_dotenv

    load_dotenv()
    # [debug]check envs
    # import os
    # environment_variables = os.environ
    # for key, value in environment_variables.items():
    #   print(f"{key}={value}")
except ImportError:
    import warnings

    warnings.warn(
        "dotenv not found. Please make sure to set your environment variables manually.",
        ImportWarning,
    )

def select_model():
    """利用するLLMをサイドバーの選択状態によって切り替える"""
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    models = ("GPT-3.5", "GPT-4", "Claude", "Gemini")
    model = st.sidebar.radio("Choose a model:", models)

    match model:
        case "GPT-3.5":
            st.session_state.model_name = "gpt-3.5-turbo"
        case "GPT-4":
            st.session_state.model_name = "gpt-4o"
        # case "Claude":
        #     return ChatOpenAI(model="claude", temperature=temperature)
        # case "Gemini":
        #     return ChatOpenAI(model="gemini", temperature=temperature)
        case _:
            raise ValueError("Invalid model selected.")

def init_page():
    """ページの基本設定"""
    st.set_page_config(
        page_title="My ChatGPT",
        page_icon="🤖",
    )
    st.header("My ChatGPT")
    st.sidebar.title("Options")

def init_messages():
    """会話履歴の消去"""
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant."),
        ]


def main():
    init_page()
    init_messages()
    llm = select_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            *st.session_state.message_history,
            ("user", "{user_input}"),
        ]
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    if user_input := st.chat_input("Ask me anything"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"{user_input}", user_input})

        st.session_state.message_history.append(("user", user_input))

        st.session_state.message_history.append(("ai", response))

    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)


if __name__ == "__main__":
    main()
