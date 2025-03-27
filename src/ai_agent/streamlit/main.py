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


def main():
    st.set_page_config(
        page_title="My ChatGPT",
        page_icon="🤖",
    )
    st.header("My ChatGPT")

    if "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant."),
        ]

    llm = ChatOpenAI(temperature=0)

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
