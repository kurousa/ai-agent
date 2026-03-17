from unittest.mock import patch, MagicMock

# We need to mock these before the import of get_content
MOCK_MODULES = {
    "streamlit": MagicMock(),
    "requests": MagicMock(),
    "bs4": MagicMock(),
    "langchain_core": MagicMock(),
    "langchain_core.prompts": MagicMock(),
    "langchain_core.output_parsers": MagicMock(),
    "langchain_openai": MagicMock(),
    "langchain_anthropic": MagicMock(),
    "langchain_google_genai": MagicMock(),
}

with patch.dict("sys.modules", MOCK_MODULES):
    import ai_agent.streamlit.website_summarizer as website_summarizer  # noqa: E402
    from ai_agent.streamlit.website_summarizer import get_content  # noqa: E402


@patch.object(website_summarizer, "st")
@patch.object(website_summarizer, "requests")
@patch.object(website_summarizer, "BeautifulSoup")
def test_get_content_fallback_to_body(mock_bs, mock_requests, mock_st):
    """<main>も<article>もない場合、<body>の内容を取得することを検証"""
    # Mock response
    mock_response = MagicMock()
    # Rationale: Mocking the HTML text returned by requests.get to contain only a <body> tag
    mock_response.text = "<html><body>Target Content</body></html>"
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    # Mock BeautifulSoup since it's mocked in sys.modules, we need to define its behavior
    mock_soup = MagicMock()
    mock_soup.main = None
    mock_soup.article = None
    mock_soup.body.get_text.return_value = "Target Content"
    mock_bs.return_value = mock_soup

    result = get_content("http://example.com/path", "93.184.216.34")

    assert result == "Target Content"
    mock_st.spinner.assert_called()
    mock_requests.get.assert_called_once_with(
        "http://93.184.216.34/path",
        headers={"Host": "example.com"},
        allow_redirects=False,
        timeout=10,
    )


@patch.object(website_summarizer, "st")
@patch.object(website_summarizer, "requests")
@patch.object(website_summarizer, "BeautifulSoup")
def test_get_content_main_priority(mock_bs, mock_requests, mock_st):
    """<main>がある場合、優先的にその内容を取得することを検証"""
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    # Mock BeautifulSoup
    mock_soup = MagicMock()
    mock_soup.main.get_text.return_value = "Main Content"
    mock_bs.return_value = mock_soup

    result = get_content("http://example.com", "93.184.216.34")

    assert result == "Main Content"


@patch.object(website_summarizer, "st")
@patch.object(website_summarizer, "requests")
@patch.object(website_summarizer, "BeautifulSoup")
def test_get_content_article_priority(mock_bs, mock_requests, mock_st):
    """<main>がなく<article>がある場合、その内容を取得することを検証"""
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    # Mock BeautifulSoup
    mock_soup = MagicMock()
    mock_soup.main = None
    mock_soup.article.get_text.return_value = "Article Content"
    mock_bs.return_value = mock_soup

    result = get_content("http://example.com", "93.184.216.34")

    assert result == "Article Content"


@patch.object(website_summarizer, "st")
@patch.object(website_summarizer, "requests")
def test_get_content_redirect_not_allowed(mock_requests, mock_st):
    """リダイレクトが発生した場合にNoneを返し、エラーを表示することを検証"""
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 301
    mock_requests.get.return_value = mock_response

    result = get_content("http://example.com", "93.184.216.34")

    assert result is None
    mock_st.error.assert_called_with("リダイレクトは許可されていません。")


@patch.object(website_summarizer, "st")
@patch.object(website_summarizer, "requests")
@patch.object(website_summarizer, "BeautifulSoup")
def test_get_content_length_limit(mock_bs, mock_requests, mock_st):
    """取得したコンテンツが長すぎる場合、切り詰められることを検証"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    mock_soup = MagicMock()
    mock_soup.main.get_text.return_value = "A" * 25000
    mock_bs.return_value = mock_soup

    result = get_content("http://example.com", "93.184.216.34")

    assert len(result) == 20000
    assert result == "A" * 20000


@patch.object(website_summarizer, "select_model")
@patch.object(website_summarizer, "ChatPromptTemplate")
@patch.object(website_summarizer, "StrOutputParser")
def test_init_chain_prompt_structure(
    mock_str_output_parser, mock_chat_prompt_template, mock_select_model
):
    """init_chainがSystemMessageとHumanMessageに分割されたプロンプトを使用しているか検証"""
    # Simply call init_chain and see how ChatPromptTemplate was initialized
    mock_select_model.return_value = MagicMock()
    mock_chat_prompt_template.from_messages.return_value = MagicMock()

    website_summarizer.init_chain()

    # Assert that ChatPromptTemplate.from_messages was called with a list containing a system and a user message
    mock_chat_prompt_template.from_messages.assert_called_once()
    args, kwargs = mock_chat_prompt_template.from_messages.call_args
    messages_list = args[0]

    assert len(messages_list) == 2
    assert messages_list[0][0] == "system"
    assert "あなたは優秀な要約アシスタントです" in messages_list[0][1]
    assert messages_list[1][0] == "user"
    assert messages_list[1][1] == "{content}"
