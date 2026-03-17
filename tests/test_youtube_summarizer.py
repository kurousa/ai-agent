from unittest.mock import patch, MagicMock
from ai_agent.utils import validate_url, validate_youtube_url


def test_validate_url_valid_youtube():
    """YouTube URLの検証（validate_urlはIPアドレスを返す）"""
    with patch("ai_agent.utils.socket.getaddrinfo") as mock:
        mock.return_value = [(2, 1, 6, "", ("142.250.185.206", 0))]
        result = validate_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result == "142.250.185.206"


def test_validate_url_valid_short_youtube():
    with patch("ai_agent.utils.socket.getaddrinfo") as mock:
        mock.return_value = [(2, 1, 6, "", ("142.250.185.206", 0))]
        result = validate_url("https://youtu.be/dQw4w9WgXcQ")
        assert result == "142.250.185.206"


def test_validate_url_valid_generic():
    with patch("ai_agent.utils.socket.getaddrinfo") as mock:
        mock.return_value = [(2, 1, 6, "", ("142.250.185.206", 0))]
        result = validate_url("https://google.com")
        assert result == "142.250.185.206"


def test_validate_url_no_scheme():
    assert validate_url("www.youtube.com/watch?v=dQw4w9WgXcQ") is None


def test_validate_url_no_netloc():
    assert validate_url("https://") is None


def test_validate_url_empty_string():
    assert validate_url("") is None


def test_validate_url_random_text():
    assert validate_url("just some text") is None


def test_validate_url_malformed():
    assert validate_url("http://[::1]]/") is None


def test_validate_youtube_url_valid():
    """validate_youtube_urlがYouTubeドメインを正しく検証すること"""
    with patch("ai_agent.utils.socket.getaddrinfo") as mock:
        mock.return_value = [(2, 1, 6, "", ("142.250.185.206", 0))]
        assert validate_youtube_url("https://www.youtube.com/watch?v=abc") is True


def test_validate_youtube_url_non_youtube():
    assert validate_youtube_url("https://evil.com/watch?v=abc") is False


MOCK_MODULES = {
    "streamlit": MagicMock(),
    "langchain_core": MagicMock(),
    "langchain_core.prompts": MagicMock(),
    "langchain_core.output_parsers": MagicMock(),
    "langchain_openai": MagicMock(),
    "langchain_anthropic": MagicMock(),
    "langchain_google_genai": MagicMock(),
    "langchain_community.document_loaders": MagicMock(),
}

with patch.dict("sys.modules", MOCK_MODULES):
    import ai_agent.streamlit.youtube_summarizer as youtube_summarizer  # noqa: E402


@patch.object(youtube_summarizer, "select_model")
@patch.object(youtube_summarizer, "ChatPromptTemplate")
def test_youtube_summarizer_prompt_structure(
    mock_chat_prompt_template, mock_select_model
):
    """Verify that the prompt correctly separates system instructions and user content to prevent prompt injection."""

    mock_select_model.return_value = MagicMock()
    mock_chat_prompt_template.from_messages.return_value = MagicMock()

    # Call the function that creates the chain (and the prompt)
    youtube_summarizer.init_chain()

    # Verify from_messages was called
    mock_chat_prompt_template.from_messages.assert_called_once()

    # Get the messages argument passed to from_messages
    messages = mock_chat_prompt_template.from_messages.call_args[0][0]

    # Verify structure: [(system, ...), (user, {content})]
    assert len(messages) == 2, "Prompt should have two messages (system and user)"

    system_role, system_content = messages[0]
    assert system_role == "system", "First message should be the system prompt"
    assert "要約" in system_content, "System prompt should contain the instructions"

    user_role, user_content = messages[1]
    assert user_role == "user", "Second message should be the user content"
    assert user_content == "{content}", (
        "User message should just be the content variable"
    )
