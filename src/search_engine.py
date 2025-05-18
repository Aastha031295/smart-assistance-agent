"""
Search engine module for retrieving information from the internet.
"""

import json
import logging
from typing import List, Dict, Any

import requests

from src.config import settings, SearchProvider

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level.value)


class SearchResult:
    """Structured search result."""
    
    def __init__(self, title: str, snippet: str, url: str):
        """
        Initialize a search result.
        
        Args:
            title: Title of the search result
            snippet: Text snippet from the result
            url: URL of the result
        """
        self.title = title
        self.snippet = snippet
        self.url = url
    
    def __repr__(self) -> str:
        return f"SearchResult(title='{self.title[:30]}...', url='{self.url}')"


class SearchEngine:
    """Search engine for retrieving information from the internet."""
    
    def __init__(self):
        """Initialize the search engine with configuration from settings."""
        self.provider = settings.search_provider
        self.api_key = settings.search_api_key.get_secret_value() if settings.search_api_key else None
        self.google_cse_id = settings.google_cse_id
        
        if not self.api_key:
            logger.warning("No search API key provided. Internet search will be simulated.")
    
    def search(self, query: str, num_results: int = 3) -> List[SearchResult]:
        """
        Search the internet for information.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        # Enhance query with car repair context
        enhanced_query = f"car repair {query}"
        
        try:
            if not self.api_key:
                return self._simulate_search(query)
            
            # Select appropriate search method based on provider
            if self.provider == SearchProvider.SERPAPI:
                return self._search_with_serpapi(enhanced_query, num_results)
            elif self.provider == SearchProvider.SERPER:
                return self._search_with_serper(enhanced_query, num_results)
            elif self.provider == SearchProvider.BRAVE:
                return self._search_with_brave(enhanced_query, num_results)
            elif self.provider == SearchProvider.GOOGLE:
                return self._search_with_google(enhanced_query, num_results)
            else:
                logger.error(f"Unsupported search provider: {self.provider}")
                return self._simulate_search(query)
                
        except Exception as e:
            logger.error(f"Error searching internet: {str(e)}")
            return self._simulate_search(query)
    
    def _search_with_serpapi(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Search using SerpAPI.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("organic_results", [])[:num_results]:
            result = SearchResult(
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                url=item.get("link", "")
            )
            results.append(result)
        
        return results
    
    def _search_with_serper(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Search using Serper.dev.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        url = "https://api.serper.dev/search"
        
        payload = json.dumps({
            "q": query
        })
        
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("organic", [])[:num_results]:
            result = SearchResult(
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                url=item.get("link", "")
            )
            results.append(result)
        
        return results
    
    def _search_with_brave(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Search using Brave Search API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        url = "https://api.search.brave.com/res/v1/web/search"
        
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }
        
        params = {
            'q': query,
            'count': num_results
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("web", {}).get("results", [])[:num_results]:
            result = SearchResult(
                title=item.get("title", ""),
                snippet=item.get("description", ""),
                url=item.get("url", "")
            )
            results.append(result)
        
        return results
    
    def _search_with_google(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Search using Google Custom Search JSON API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        url = "https://www.googleapis.com/customsearch/v1"
        
        params = {
            'key': self.api_key,
            'cx': self.google_cse_id,
            'q': query,
            'num': min(num_results, 10)  # Google CSE has a max of 10 results per query
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("items", [])[:num_results]:
            result = SearchResult(
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                url=item.get("link", "")
            )
            results.append(result)
        
        return results
    
    def _simulate_search(self, query: str) -> List[SearchResult]:
        """
        Simulate search results for demonstration purposes.
        
        Args:
            query: Search query
            
        Returns:
            List of simulated search results
        """
        logger.info(f"Simulating search for: {query}")
        
        # Return simulated search results based on query keywords
        if "headlight" in query.lower() or "headlamp" in query.lower():
            return [
                SearchResult(
                    title="How to Fix Headlights That Don't Work - Car Maintenance Guide",
                    snippet="If your headlights aren't working, first check the bulbs, then fuses, wiring connections, and finally the headlight switch and relay. Modern LED headlights may also have control modules that can fail.",
                    url="https://example.com/headlight-repair"
                ),
                SearchResult(
                    title="Diagnosing Headlight Problems - AutoRepair",
                    snippet="Common causes of headlight failure include blown bulbs, corroded sockets, wiring issues, faulty relays, and bad grounding. LED headlights may also suffer from driver module failures.",
                    url="https://example.com/headlight-diagnosis"
                )
            ]
        elif "brake" in query.lower():
            return [
                SearchResult(
                    title="When to Replace Brake Pads - Car Maintenance",
                    snippet="Signs you need new brake pads: squeaking/grinding noises, taking longer to stop, brake pedal pulsation, visible wear (less than 1/4 inch pad), and dashboard warning light.",
                    url="https://example.com/brake-replacement"
                ),
                SearchResult(
                    title="DIY Brake Pad Replacement - Step-by-Step Guide",
                    snippet="To replace brake pads: 1) Loosen lug nuts, 2) Jack up car and secure, 3) Remove wheels, 4) Remove caliper bolts, 5) Pivot caliper away, 6) Remove old pads, 7) Compress caliper piston, 8) Install new pads, 9) Reinstall caliper, 10) Reinstall wheels.",
                    url="https://example.com/diy-brake-replacement"
                )
            ]
        else:
            # Generic car problem results
            return [
                SearchResult(
                    title="Common Car Problems and Solutions - Auto Repair Guide",
                    snippet="Frequent car issues include dead batteries, flat tires, overheating engines, brake problems, and check engine lights. Regular maintenance helps prevent most issues before they become serious.",
                    url="https://example.com/car-problems"
                ),
                SearchResult(
                    title="DIY Car Repair: When to Fix It Yourself vs. Visit a Mechanic",
                    snippet="Simple repairs like changing air filters, replacing wiper blades, and swapping bulbs can be DIY. Leave complex jobs like transmission work, timing belt replacement, and engine repairs to professionals.",
                    url="https://example.com/diy-vs-mechanic"
                )
            ]


# Create global search engine instance
search_engine = SearchEngine()
