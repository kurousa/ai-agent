import socket
import ipaddress
from urllib.parse import urlparse

def validate_url(url):
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        if result.scheme not in ["http", "https"]:
            return False

        hostname = result.hostname
        if not hostname:
            return False

        # Resolve hostname to IP addresses and check each one
        addr_info = socket.getaddrinfo(hostname, None)
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
