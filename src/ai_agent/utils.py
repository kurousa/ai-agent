import ipaddress
import socket
from functools import lru_cache
from urllib.parse import urlparse

# YouTube で許可されるドメインのホワイトリスト
YOUTUBE_ALLOWED_DOMAINS = {
    "www.youtube.com",
    "youtube.com",
    "youtu.be",
    "m.youtube.com",
}

# 最大画像サイズ (MB)
MAX_IMAGE_SIZE_MB = 5
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


def is_file_size_valid(file_size: int) -> bool:
    """ファイルサイズが制限内であるかを確認する。

    Args:
        file_size: ファイルサイズ（バイト）

    Returns:
        bool: 制限内であればTrue、そうでなければFalse
    """
    return file_size <= MAX_IMAGE_SIZE_BYTES


@lru_cache(maxsize=128)
def _getaddrinfo_cached(hostname):
    """Resolve hostname to IP addresses with caching."""
    return socket.getaddrinfo(hostname, None)


def validate_url(url):
    """URLを検証し、安全であれば解決済みIPアドレスを返す。

    TOCTOU対策: 検証済みのIPアドレスを返すことで、
    呼び出し元が同じIPに対して直接リクエストを送信可能。

    Args:
        url: 検証対象のURL文字列

    Returns:
        str: 安全な解決済みIPアドレス（検証成功時）
        None: 検証失敗時
    """
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return None
        if result.scheme not in ["http", "https"]:
            return None

        hostname = result.hostname
        if not hostname:
            return None

        # Resolve hostname to IP addresses and check each one
        addr_info = _getaddrinfo_cached(hostname)
        for _, _, _, _, sockaddr in addr_info:
            ip_addr = sockaddr[0]
            ip = ipaddress.ip_address(ip_addr)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_unspecified
                or ip.is_multicast
                or ip.is_reserved
            ):
                return None

        # Return the first resolved safe IP address
        first_ip = addr_info[0][4][0]
        return first_ip
    except (ValueError, socket.gaierror):
        return None


def validate_youtube_url(url):
    """YouTube URLを検証する。

    YouTubeドメインのホワイトリストで検証し、
    さらにIPアドレスの安全性も確認する。

    Args:
        url: 検証対象のURL文字列

    Returns:
        bool: 検証成功時True、失敗時False
    """
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        if result.scheme not in ["http", "https"]:
            return False

        hostname = result.hostname
        if not hostname:
            return False

        # YouTube ドメインのホワイトリスト検証
        if hostname not in YOUTUBE_ALLOWED_DOMAINS:
            return False

        # IPアドレスの安全性も確認
        addr_info = _getaddrinfo_cached(hostname)
        for _, _, _, _, sockaddr in addr_info:
            ip_addr = sockaddr[0]
            ip = ipaddress.ip_address(ip_addr)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_unspecified
                or ip.is_multicast
                or ip.is_reserved
            ):
                return False

        return True
    except (ValueError, socket.gaierror):
        return False
