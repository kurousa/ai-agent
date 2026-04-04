import timeit
import socket
from unittest.mock import patch
from ai_agent.utils import validate_youtube_url

# Setup mock for getaddrinfo to avoid actual network calls and make it consistent
def mock_getaddrinfo(hostname, port, family=0, type=0, proto=0, flags=0):
    if any(domain in hostname for domain in ["youtube.com", "youtu.be"]):
        return [(2, 1, 6, "", ("142.250.185.206", 0))]
    return [(2, 1, 6, "", ("93.184.216.34", 0))]

def run_benchmark():
    test_cases = [
        ("https://www.youtube.com/watch?v=abc123", True),
        ("https://youtu.be/abc123", True),
        ("https://m.youtube.com/watch?v=abc123", True),
        ("https://evil.com/watch?v=abc123", False),
        ("invalid-url", False),
        ("ftp://www.youtube.com/watch?v=abc123", False),
    ]

    with patch("socket.getaddrinfo", side_effect=mock_getaddrinfo):
        # Verification
        print("Verifying correctness...")
        for url, expected in test_cases:
            actual = validate_youtube_url(url)
            print(f"URL: {url:50} Expected: {expected!s:5} Actual: {actual!s:5}")
            assert actual == expected, f"Validation failed for {url}"
        print("Verification passed!\n")

        urls = [tc[0] for tc in test_cases]

        def test_func():
            for url in urls:
                validate_youtube_url(url)

        timer = timeit.Timer(test_func)
        number = 20000
        result = timer.timeit(number=number)
        print(f"Time for {number} iterations: {result:.4f} seconds")
        print(f"Average time per call: {result / (number * len(urls)) * 1e6:.4f} microseconds")

if __name__ == "__main__":
    run_benchmark()
