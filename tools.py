#!/usr/bin/env python3
"""
Tools for the voice assistant.

This module contains various tools used by the voice assistant, such as:
- WebBrowser: A simple web browser implementation for web searches
"""

# Standard library imports
import logging
import re

# Third-party imports
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WebBrowser:
    """A simple web browser implementation inspired by smolagents."""
    
    def __init__(self):
        """Initialize the web browser."""
        self.current_url = None
        self.current_page_content = None
        self.history = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def navigate(self, url):
        """Navigate to a URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            Dictionary with status and content
        """
        logger.info(f"Navigating to: {url}")
        
        try:
            # Add http:// if not present
            if not url.startswith('http'):
                url = 'https://' + url
            
            # Make the request
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Update state
            self.current_url = url
            self.current_page_content = response.text
            self.history.append(url)
            
            # Extract title
            title = self._extract_title(response.text)
            
            return {
                'status': 'success',
                'url': url,
                'title': title,
                'content_length': len(response.text)
            }
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            return {
                'status': 'error',
                'url': url,
                'error': str(e)
            }
    
    def search(self, query):
        """Search the web for a query.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        logger.info(f"Searching for: {query}")
        
        try:
            # Use DuckDuckGo for searching
            search_url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            # Make the request
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Extract results
            results = []
            
            # Add the abstract if available
            if data.get('Abstract') and data.get('AbstractURL'):
                results.append({
                    'title': data.get('Heading', 'Abstract'),
                    'url': data.get('AbstractURL'),
                    'snippet': data.get('Abstract')
                })
            
            # Add related topics
            if data.get('RelatedTopics'):
                for topic in data.get('RelatedTopics'):
                    if 'Text' in topic and 'FirstURL' in topic:
                        results.append({
                            'title': topic.get('Text').split(' - ')[0] if ' - ' in topic.get('Text') else topic.get('Text'),
                            'url': topic.get('FirstURL'),
                            'snippet': topic.get('Text')
                        })
            
            return {
                'status': 'success',
                'query': query,
                'results': results[:5]  # Limit to 5 results
            }
        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return {
                'status': 'error',
                'query': query,
                'error': str(e)
            }
    
    def extract_content(self, selector=None):
        """Extract content from the current page.
        
        Args:
            selector: CSS selector to extract content from
            
        Returns:
            Dictionary with extracted content
        """
        if not self.current_page_content:
            return {
                'status': 'error',
                'error': 'No page loaded'
            }
        
        try:
            # Simple extraction without BeautifulSoup
            if selector:
                # Very basic extraction - in a real implementation, use BeautifulSoup
                content = f"Content extraction with selector '{selector}' not implemented"
            else:
                # Extract text content (simplified)
                content = self._extract_text(self.current_page_content)
            
            return {
                'status': 'success',
                'url': self.current_url,
                'content': content[:1000]  # Limit content length
            }
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return {
                'status': 'error',
                'url': self.current_url,
                'error': str(e)
            }
    
    def _extract_title(self, html):
        """Extract title from HTML.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted title or default title
        """
        try:
            # Simple regex to extract title
            title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
            if title_match:
                return title_match.group(1).strip()
            return "Untitled Page"
        except Exception:
            return "Untitled Page"
    
    def _extract_text(self, html):
        """Extract text content from HTML.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted text content
        """
        try:
            # Simple regex to remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception:
            return "Failed to extract text content" 