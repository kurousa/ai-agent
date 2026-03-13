from unittest.mock import patch
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
