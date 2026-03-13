from unittest.mock import patch
from ai_agent.utils import validate_url, validate_youtube_url


class TestValidateUrl:
    """validate_url のテスト"""

    @patch("ai_agent.utils.socket.getaddrinfo")
    def test_valid_public_url_returns_ip(self, mock_getaddrinfo):
        """安全な公開URLに対してIPアドレス文字列を返すこと"""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]
        result = validate_url("https://example.com")
        assert result == "93.184.216.34"

    @patch("ai_agent.utils.socket.getaddrinfo")
    def test_private_ip_returns_none(self, mock_getaddrinfo):
        """プライベートIPに解決されるURLにはNoneを返すこと"""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("192.168.1.1", 0)),
        ]
        result = validate_url("https://internal.example.com")
        assert result is None

    @patch("ai_agent.utils.socket.getaddrinfo")
    def test_loopback_ip_returns_none(self, mock_getaddrinfo):
        """ループバックIPに解決されるURLにはNoneを返すこと"""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]
        result = validate_url("https://localhost.example.com")
        assert result is None

    @patch("ai_agent.utils.socket.getaddrinfo")
    def test_link_local_ip_returns_none(self, mock_getaddrinfo):
        """リンクローカルIPに解決されるURLにはNoneを返すこと"""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("169.254.1.1", 0)),
        ]
        result = validate_url("https://linklocal.example.com")
        assert result is None

    def test_invalid_scheme_returns_none(self):
        """http/https以外のスキームにはNoneを返すこと"""
        assert validate_url("ftp://example.com") is None
        assert validate_url("file:///etc/passwd") is None

    def test_no_scheme_returns_none(self):
        """スキームなしURLにはNoneを返すこと"""
        assert validate_url("example.com") is None

    def test_empty_string_returns_none(self):
        """空文字にはNoneを返すこと"""
        assert validate_url("") is None

    def test_returns_string_type(self):
        """戻り値がstr型またはNoneであること"""
        result = validate_url("")
        assert result is None

    @patch("ai_agent.utils.socket.getaddrinfo")
    def test_multiple_ips_with_one_private_returns_none(self, mock_getaddrinfo):
        """複数IP解決のうち1つでもプライベートならNoneを返すこと"""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0)),
            (2, 1, 6, "", ("10.0.0.1", 0)),
        ]
        result = validate_url("https://mixed.example.com")
        assert result is None


class TestValidateYoutubeUrl:
    """validate_youtube_url のテスト"""

    @patch("ai_agent.utils.socket.getaddrinfo")
    def test_valid_youtube_url(self, mock_getaddrinfo):
        """正規YouTubeURLはTrueを返すこと"""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("142.250.185.206", 0)),
        ]
        assert validate_youtube_url("https://www.youtube.com/watch?v=abc123") is True

    @patch("ai_agent.utils.socket.getaddrinfo")
    def test_valid_youtu_be_url(self, mock_getaddrinfo):
        """短縮YouTubeURLはTrueを返すこと"""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("142.250.185.206", 0)),
        ]
        assert validate_youtube_url("https://youtu.be/abc123") is True

    def test_non_youtube_domain_returns_false(self):
        """YouTube以外のドメインはFalseを返すこと"""
        assert validate_youtube_url("https://evil.com/watch?v=abc123") is False

    def test_invalid_scheme_returns_false(self):
        """不正スキームはFalseを返すこと"""
        assert validate_youtube_url("ftp://www.youtube.com/watch?v=abc123") is False

    def test_empty_url_returns_false(self):
        """空URLはFalseを返すこと"""
        assert validate_youtube_url("") is False
