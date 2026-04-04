from unittest.mock import MagicMock, patch

# Mocking tiktoken as well
mock_tiktoken = MagicMock()
mock_st = MagicMock()


# Using a class to handle session_state more realistically
class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return MagicMock()

    def __setattr__(self, key, value):
        self[key] = value


mock_st.session_state = SessionState()

MOCK_MODULES = {
    "streamlit": mock_st,
    "tiktoken": mock_tiktoken,
    "langchain_openai": MagicMock(),
    "langchain_anthropic": MagicMock(),
    "langchain_google_genai": MagicMock(),
    "langchain_core.prompts": MagicMock(),
    "langchain_core.output_parsers": MagicMock(),
}

with patch.dict("sys.modules", MOCK_MODULES):
    import ai_agent.streamlit.chat as chat  # noqa: E402


def test_get_message_counts_claude():
    """Claudeモデルの場合にllm.get_num_tokensが呼ばれることを確認"""
    mock_st.session_state.model_name = "claude-3-5-haiku-20241022"
    mock_st.session_state.token_count_cache = {}
    mock_llm = MagicMock()
    mock_st.session_state.llm = mock_llm
    mock_llm.get_num_tokens.return_value = 3

    result = chat.get_message_counts("hello")

    assert result == 3
    mock_llm.get_num_tokens.assert_called_once_with("hello")


def test_get_message_counts_gpt():
    """GPTモデルの場合にllm.get_num_tokensが呼ばれることを確認"""
    mock_st.session_state.model_name = "gpt-4o"
    mock_st.session_state.token_count_cache = {}
    mock_llm = MagicMock()
    mock_st.session_state.llm = mock_llm
    mock_llm.get_num_tokens.return_value = 4

    result = chat.get_message_counts("hello world")

    assert result == 4
    mock_llm.get_num_tokens.assert_called_once_with("hello world")


def test_get_message_counts_gemini():
    """Geminiモデルの場合にllm.get_num_tokensが呼ばれることを確認"""
    mock_st.session_state.model_name = "gemini-1.5-flash-latest"
    mock_st.session_state.token_count_cache = {}
    mock_llm = MagicMock()
    mock_st.session_state.llm = mock_llm
    mock_llm.get_num_tokens.return_value = 5

    result = chat.get_message_counts("test message")

    assert result == 5
    mock_llm.get_num_tokens.assert_called_once_with("test message")


def test_get_message_counts_caching():
    """同じメッセージに対してget_message_countsが呼ばれた際、キャッシュが使用されることを確認"""
    mock_st.session_state.model_name = "gemini-1.5-flash-latest"
    mock_llm = MagicMock()
    mock_st.session_state.llm = mock_llm
    mock_llm.get_num_tokens.return_value = 10
    mock_st.session_state.token_count_cache = {}

    # 1回目の呼び出し
    result1 = chat.get_message_counts("cached message")
    assert result1 == 10
    assert mock_llm.get_num_tokens.call_count == 1

    # 2回目の呼び出し（同じメッセージ）
    result2 = chat.get_message_counts("cached message")
    assert result2 == 10
    # 呼び出し回数が増えていないことを確認
    assert mock_llm.get_num_tokens.call_count == 1
    assert (
        mock_st.session_state.token_count_cache[
            ("gemini-1.5-flash-latest", "cached message")
        ]
        == 10
    )
