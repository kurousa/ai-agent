import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Mock streamlit and requests before importing the target module
sys.modules['streamlit'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.prompts'] = MagicMock()
sys.modules['langchain_core.output_parsers'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_anthropic'] = MagicMock()
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['bs4'] = MagicMock()

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ai_agent.streamlit.website_summarizer import get_content

class TestWebsiteSummarizer(unittest.TestCase):
    @patch('ai_agent.streamlit.website_summarizer.requests.get')
    def test_get_content_timeout(self, mock_get):
        import requests
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Content</body></html>"
        mock_get.return_value = mock_response

        # Mock BeautifulSoup
        with patch('ai_agent.streamlit.website_summarizer.BeautifulSoup') as mock_bs:
            mock_soup = MagicMock()
            mock_soup.main = None
            mock_soup.article = None
            mock_soup.body.get_text.return_value = "Content"
            mock_bs.return_value = mock_soup

            # Call the function
            url = "http://example.com"
            safe_ip = "93.184.216.34"
            get_content(url, safe_ip)

            # Check if requests.get was called with timeout=10
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            self.assertEqual(kwargs.get('timeout'), 10)

if __name__ == '__main__':
    unittest.main()
