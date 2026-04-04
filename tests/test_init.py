from ai_agent import hello


def test_hello():
    """hello() が正しい文字列を返すこと"""
    assert hello() == "Hello from ai-agent!"
