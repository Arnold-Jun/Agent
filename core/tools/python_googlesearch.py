from typing import Any, Optional, List, Dict
import dspy
from dspy import Prediction
import requests
from bs4 import BeautifulSoup
import concurrent.futures

try:
    from googlesearch import search
except ImportError:
    raise ImportError("`googlesearch-python` not installed. Please install using `pip install googlesearch-python`")

try:
    from pycountry import pycountry
except ImportError:
    raise ImportError("`pycountry` not installed. Please install using `pip install pycountry`")

class GoogleSearch(dspy.Module):
    """
   Search for information from Google Chrome. It will be called when user want to search something oneline.
   Specially when user ask some question which is not related to DKU or Duke.

    """

    def __init__(
            self,
            fixed_max_results: Optional[int] = None,
            fixed_language: Optional[str] = None,
            headers: Optional[Any] = None,
            proxy: Optional[str] = None,
            timeout: Optional[int] = 20,
            enable_content: bool = True,
            content_length: int = 1000
    ):
        super().__init__()

        # 原有参数...
        self.enable_content = enable_content
        self.content_length = content_length
        self.fixed_max_results: Optional[int] = fixed_max_results
        self.fixed_language: Optional[str] = fixed_language
        self.headers: Optional[Any] = headers
        self.proxy: Optional[str] = proxy
        self.timeout: Optional[int] = timeout

    def _fetch_web_content(self, url: str) -> Dict[str, str]:
        """Get the core content of the web page"""
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()

            # Parse using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            content = ""

            # Try to find the body tag
            main_content = soup.find('article') or soup.find('main')
            if main_content:
                paragraphs = main_content.find_all(['p', 'h2', 'h3'])
            else:
                paragraphs = soup.find_all(['p', 'h2', 'h3'])

            # Filter out useless content such as ads
            blacklist = ['footer', 'nav', 'sidebar', 'ads']
            for p in paragraphs:
                if any(tag in p.parent.get('class', []) for tag in blacklist):
                    continue
                content += p.get_text().strip() + "\n"

            # 智能截取关键段落
            content = ' '.join(content.split()[:self.content_length])
            return {"content": content, "error": None}

        except Exception as e:
            return {"content": "", "error": str(e)}

    def forward(self, query: str, internal_memory: {}, max_results: int = 3, language: str = "en") -> Prediction:
        """Enhanced Google Search"""
        if "search_url" not in internal_memory:
            internal_memory["search_url"] = []

        max_results = self.fixed_max_results or max_results
        language = self.fixed_language or language

        # Resolve language to ISO 639-1 code if needed
        if len(language) != 2:
            _language = pycountry.languages.lookup(language)
            if _language:
                language = _language.alpha_2
            else:
                language = "en"

        # Perform Google search using the googlesearch-python package
        try:
            results = list(search(query, num_results=max_results, lang=language, advanced=True))
        except:
            return dspy.Prediction(
                result=[{"text":  "Your computer failed to request access to google. Maybe the network can't connect to google."
                                  "Please check the status of your computer network and then try again"}],
                internal_result={}
            )

        results = [
            r for r in results
            if r.url not in internal_memory["search_url"]
        ]

        # If there are no search results, they are returned early
        if not results:
            return dspy.Prediction(
                result=[{"text": "No search results from Google. You can search by yourself."},],
                internal_result={}
            )

        # Fetch web content in parallel
        if self.enable_content:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_url = {executor.submit(self._fetch_web_content, result.url): result.url for result in results}
                contents = {}
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    contents[url] = future.result()

        enhanced_res = []
        processed_urls = []

        for result in results:
            content_data = contents.get(result.url, {})

            enhanced_res.append({
                "url": result.url,
                "text": content_data.get("content", "")
            })

            processed_urls.append(result.url)

        internal_memory["search_url"].extend(processed_urls)

        return dspy.Prediction(
            result=enhanced_res,
            internal_result={"search_url": processed_urls}
        )