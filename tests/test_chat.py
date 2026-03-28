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
    """Claudeモデルの場合にcl100k_baseが使用されることを確認"""
    mock_st.session_state.model_name = "claude-3-5-haiku-20241022"
    mock_st.session_state.token_count_cache = {}

    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3]  # 3 tokens
    mock_tiktoken.get_encoding.return_value = mock_encoding

    # Reset mocks to clear any calls during import or previous tests
    mock_tiktoken.get_encoding.reset_mock()
    mock_tiktoken.encoding_for_model.reset_mock()

    result = chat.get_message_counts("hello")

    assert result == 3
    mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
    mock_tiktoken.encoding_for_model.assert_not_called()
    mock_encoding.encode.assert_called_once_with("hello")


def test_get_message_counts_gpt():
    """GPTモデルの場合にモデル名に基づいたエンコーディングが使用されることを確認"""
    mock_st.session_state.model_name = "gpt-4o"
    mock_st.session_state.token_count_cache = {}

    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3, 4]  # 4 tokens
    mock_tiktoken.encoding_for_model.return_value = mock_encoding

    # Reset mocks
    mock_tiktoken.get_encoding.reset_mock()
    mock_tiktoken.encoding_for_model.reset_mock()

    result = chat.get_message_counts("hello world")

    assert result == 4
    mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4o")
    mock_tiktoken.get_encoding.assert_not_called()
    mock_encoding.encode.assert_called_once_with("hello world")


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


def test_get_message_counts_with_provided_encoding():
    """エンコーディングが引数で渡された場合にそれが使用されることを確認"""
    mock_st.session_state.model_name = "gpt-4o"
    mock_st.session_state.token_count_cache = {}
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2]

    # Reset mocks
    mock_tiktoken.get_encoding.reset_mock()
    mock_tiktoken.encoding_for_model.reset_mock()

    result = chat.get_message_counts("hi", encoding=mock_encoding)

    assert result == 2
    mock_encoding.encode.assert_called_once_with("hi")
    mock_tiktoken.encoding_for_model.assert_not_called()
    mock_tiktoken.get_encoding.assert_not_called()


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
