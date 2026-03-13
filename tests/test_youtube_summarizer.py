import pytest
from ai_agent.streamlit.utils import validate_url

def test_validate_url_valid_youtube():
    assert validate_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

def test_validate_url_valid_short_youtube():
    assert validate_url("https://youtu.be/dQw4w9WgXcQ") is True

def test_validate_url_valid_generic():
    assert validate_url("https://google.com") is True

def test_validate_url_no_scheme():
    assert validate_url("www.youtube.com/watch?v=dQw4w9WgXcQ") is False

def test_validate_url_no_netloc():
    assert validate_url("https://") is False

def test_validate_url_empty_string():
    assert validate_url("") is False

def test_validate_url_random_text():
    assert validate_url("just some text") is False

def test_validate_url_malformed():
    # urlparse might not raise ValueError for simple strings, but let's check
    assert validate_url("http://[::1]]/") is False
