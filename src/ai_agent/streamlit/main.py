import tiktoken
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
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

# 1_000_000トークン辺りのコストを算出する
PER_1_000_000_TOKENS = 1_000_000
MODEL_PRICE = {
    "input": {
        "gpt-3.5-turbo": 0.5 / PER_1_000_000_TOKENS,
        "gpt-4o": 5 / PER_1_000_000_TOKENS,
        "gemini-1.5-flash-latest": 0.125 / PER_1_000_000_TOKENS,
        "claude-3-5-haiku-20241022": 3 / PER_1_000_000_TOKENS
    },
    "output": {
        "gpt-3.5-turbo": 1.5 / PER_1_000_000_TOKENS,
        "gpt-4o": 15 / PER_1_000_000_TOKENS,
        "gemini-1.5-flash-latest": 0.375 / PER_1_000_000_TOKENS,
        "claude-3-5-haiku-20241022": 15 / PER_1_000_000_TOKENS
    }
}
GEMINI_PRICE_THRESHOLD_TOKENS = 128_000 # 128kトークン以上の場合、単価が変わるため

def get_message_counts(text):
    if "gemini" in st.session_state.model_name:
        return st.session_state.llm.get_num_tokens(text)
    else:
        if "gpt" in st.session_state.model_name:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        else:
            #NOTE: Claudeはトークン数を取得する方法が不明なため、1トークン=1文字として計算
            encoding = tiktoken.get_encoding("cl100k_base") # GPT models use cl100k_base encoding
            print("警告: Claude トークンの計算は近似値です。")
        return len(encoding.encode(text))

def calc_cost():
    if len(st.session_state.message_history) == 1:
        # 初期状態はシステムメッセージのみが入った状態であるため処理をスキップ
        return

    output_count = 0
    input_count = 0
    for role, message in st.session_state.message_history:
        token_count = get_message_counts(message)
        match role:
            case "user":
                input_count += token_count
            case "ai":
                output_count += token_count
            case _:
                pass

    input_cost = MODEL_PRICE["input"][st.session_state.model_name] * input_count
    output_cost = MODEL_PRICE["output"][st.session_state.model_name] * output_count
    if (
        "gemini" in st.session_state.model_name
        and ( input_count + output_count ) > GEMINI_PRICE_THRESHOLD_TOKENS
    ):
        # Geminiは、トークン数が128kトークンを超過する場合、単価が以下のように変わる仕様
        # Ref: https://ai.google.dev/gemini-api/docs/pricing#gemini-1.5-flash
        # As of 2025-04, the pricing for Gemini 1.5 Flash is as follows:
        # Input: $0.075, prompts <= 128k tokens, $0.15, prompts > 128k tokens
        # Output: $0.30, prompts <= 128k tokens, $0.60, prompts > 128k tokens
        input_cost *= 2
        output_cost *= 2
    cost = output_cost + input_cost
    return cost, output_cost, input_cost

def display_cost(cost, output_cost, input_cost):
    """サイドバーにコストを表示する"""
    st.sidebar.markdown("### Cost(USD)")
    st.sidebar.markdown(f"**Total Cost:** ${cost:.6f}")
    st.sidebar.markdown(f"**Output Cost:** ${output_cost:.6f}")
    st.sidebar.markdown(f"**Input Cost:** ${input_cost:.6f}")


def select_model():
    """利用するLLMをサイドバーの選択状態によって切り替える"""
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    models = (
        "Open AI GPT-3.5-turbo",
        "Open AI GPT-4o",
        "Claude 3.5 Haiku",
        # "Claude 3.7 Sonnet",
        "Google Gemini 1.5 Flash",
        # "Gemini 2.0 Flash"
    )
    model = st.sidebar.radio("Choose a model:", models)

    match model:
        case "Open AI GPT-3.5-turbo":
            st.session_state.model_name = "gpt-3.5-turbo"
            return ChatOpenAI(model=st.session_state.model_name, temperature=temperature)
        case "Open AI GPT-4o":
            st.session_state.model_name = "gpt-4o"
            return ChatOpenAI(model=st.session_state.model_name, temperature=temperature)
        case "Claude 3.5 Haiku":
            st.session_state.model_name = "claude-3-5-haiku-20241022"
            return ChatAnthropic(model=st.session_state.model_name, temperature=temperature)
        # case "Claude 3.7 Sonnet":
        #     st.session_state.model_name = "claude-3-7-sonnet-20250219"
        #     return ChatAnthropic(model=st.session_state.model_name, temperature=temperature)
        case "Google Gemini 1.5 Flash":
            st.session_state.model_name = "gemini-1.5-flash-latest"
            return ChatGoogleGenerativeAI(model=st.session_state.model_name, temperature=temperature)
        # case "Gemini 2.0 Flash":
        #     st.session_state.model_name = "gemini-2.0-flash-latest"
        #     return ChatGoogleGenerativeAI(model=st.session_state.model_name, temperature=temperature)
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

    # メッセージ履歴を出力
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

    # ユーザーから入力があった場合、streamにて回答を表示し、履歴に記録
    if user_input := st.chat_input("Ask me anything"):
        st.chat_message("user").markdown(user_input)

        with st.chat_message("ai"):
            response = st.write_stream(chain.stream({"user_input": user_input}))

        st.session_state.message_history.append(("user", user_input))

        st.session_state.message_history.append(("ai", response))

    # コスト計算を行い、サイドバーに出力
    cost_results = calc_cost()
    if cost_results:
        display_cost(*cost_results)

if __name__ == "__main__":
    main()
