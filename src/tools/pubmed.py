import requests
import xml.etree.ElementTree as ET
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from Bio import Entrez
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt
from utils.ml_logging import get_logger

logger = get_logger()

class PubMedScraper:
    def __init__(self, email: str):
        self.email = email
        Entrez.email = email
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

    def fetch_pubmed_ids(
        self,
        query: str,
        max_results: int = 10,
        field: Optional[str] = None,
        sort: str = "relevance",
        datetype: Optional[str] = None,
        mindate: Optional[str] = None,
        maxdate: Optional[str] = None,
        retstart: int = 0
    ) -> List[str]:
        """
        Fetch PubMed IDs based on the query.

        :param query: Search query string.
        :param max_results: Maximum number of results to return.
        :param field: (optional) Specific field to search.
        :param sort: Sort order of results.
        :param datetype: (optional) Type of date for filtering.
        :param mindate: (optional) Minimum date for filtering.
        :param maxdate: (optional) Maximum date for filtering.
        :param retstart: Starting point for results retrieval.
        :return: List of PubMed IDs.
        :raises ValueError: If the response status code is not 200.
        """
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "retmode": "json",
            "retstart": retstart
        }

        if field:
            params["field"] = field
        if datetype:
            params["datetype"] = datetype
        if mindate:
            params["mindate"] = mindate
        if maxdate:
            params["maxdate"] = maxdate

        response = requests.get(base_url, params=params, headers=self.headers)
        if response.status_code != 200:
            logger.error(f"Failed to fetch PubMed IDs: {response.status_code}")
            raise ValueError(f"Error fetching PubMed IDs: {response.status_code}")
        data = response.json()
        return data.get('esearchresult', {}).get('idlist', [])

    def fetch_pubmed_articles(self, ids: List[str]) -> str:
        """
        Fetch PubMed articles based on IDs.

        :param ids: List of PubMed IDs.
        :return: XML data of fetched articles.
        :raises ValueError: If there's an error fetching articles.
        """
        try:
            handle = Entrez.efetch("pubmed", id=",".join(ids), retmode="xml")
            return handle.read()
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            raise ValueError(f"Error fetching articles: {e}")

    def parse_article_details(self, xml_data: str) -> List[Dict[str, Any]]:
        """
        Parse article details from XML data.

        :param xml_data: XML data of articles.
        :return: List of dictionaries with article details.
        """
        root = ET.fromstring(xml_data)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", default="No PMID available")
            year = article.findtext(".//PubDate/Year", default="No Year available")
            volume = article.findtext(".//JournalIssue/Volume", default="No Volume available")
            issue = article.findtext(".//JournalIssue/Issue", default="No Issue available")
            citation = f"{year};{volume}({issue})"

            details = {
                'pmid': pmid,
                'title': article.findtext(".//ArticleTitle", default="No title available"),
                'abstract': article.findtext(".//AbstractText", default="No abstract available"),
                'authors': [
                    f"{author.findtext('ForeName', '')} {author.findtext('LastName', '')}".strip()
                    for author in article.findall(".//Author")
                ],
                'year': year,
                'volume': volume,
                'issue': issue,
                'journal': article.findtext(".//Journal/Title", default="No journal available"),
                'citation': citation,
                'link': article.findtext(".//ArticleId[@IdType='doi']", default="No link available")
            }
            if details['link'] and details['link'] != "No link available":
                details['link'] = f"https://doi.org/{details['link']}"

            pmc_id = None
            for other_id in article.findall(".//ArticleIdList/ArticleId"):
                if other_id.attrib.get('IdType') == 'pmc':
                    pmc_id = other_id.text
                    break
            if pmc_id:
                details['pdf_link'] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
            else:
                details['pdf_link'] = "No PDF link available"

            articles.append(details)
        return articles

    def articles_to_json(self, articles: List[Dict[str, Any]]) -> str:
        """
        Convert articles to JSON format.

        :param articles: List of articles.
        :return: JSON string of articles.
        """
        return json.dumps(articles, indent=4)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def fetch_pdf_content(self, url: str) -> Optional[str]:
        """
        Fetch PDF content from a given URL with retry logic.

        :param url: URL to the PDF file.
        :return: Text content of the PDF or an error message.
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            with open('temp_article.pdf', 'wb') as f:
                f.write(response.content)

            pdf_document = fitz.open('temp_article.pdf')
            text = ""
            for page in pdf_document:
                text += page.get_text()
            return text
        except requests.HTTPError as e:
            logger.error(f"Error fetching PDF content: {e}")
            return f"Error fetching PDF content: {e}"

    def add_article_content(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add full article content to the articles.

        :param articles: List of articles.
        :return: List of articles with full content added.
        """
        for article in articles:
            if 'pdf_link' in article and article['pdf_link'] != "No PDF link available":
                article['full_content'] = self.fetch_pdf_content(article['pdf_link'])
            else:
                article['full_content'] = "No full text link available"
        return articles

    def json_to_csv(self, json_data: str, csv_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Convert JSON data to a pandas DataFrame and optionally save as CSV.

        :param json_data: JSON string of articles.
        :param csv_file_path: (optional) Path to save the CSV file.
        :return: DataFrame containing the articles.
        """
        articles = json.loads(json_data)
        columns = [
            'pmid', 'title', 'abstract', 'authors', 'year', 'volume', 
            'issue', 'journal', 'citation', 'link', 'pdf_link', 'full_content'
        ]
        df = pd.DataFrame(articles, columns=columns)
        if csv_file_path:
            df.to_csv(csv_file_path, index=False)
        return df

    def search_by_query(
        self,
        query: str,
        max_results: int = 10,
        field: Optional[str] = None,
        sort: str = "relevance",
        datetype: Optional[str] = None,
        mindate: Optional[str] = None,
        maxdate: Optional[str] = None,
        retstart: int = 0,
        csv_file_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Searches PubMed based on a query and exports the results to a pandas DataFrame.

        :param query: Search query.
        :param max_results: Maximum number of results to fetch. Default is 10.
        :param field: (optional) Specific field to search.
        :param sort: Sort order of the results. Default is "relevance".
        :param datetype: (optional) Type of date for filtering.
        :param mindate: (optional) Minimum date for filtering results.
        :param maxdate: (optional) Maximum date for filtering results.
        :param retstart: Starting point for results retrieval. Default is 0.
        :param csv_file_path: (optional) Path to save the results as a CSV file.
        :return: DataFrame containing the search results.
        :raises ValueError: If no articles are found.
        """
        logger.info(f"Starting search with query: {query}")
        pubmed_ids = self.fetch_pubmed_ids(query, max_results, field, sort, datetype, mindate, maxdate, retstart)

        if not pubmed_ids:
            logger.warning("No articles found for the given query.")
            raise ValueError("No articles found for the given query.")

        xml_data = self.fetch_pubmed_articles(pubmed_ids)
        articles = self.parse_article_details(xml_data)
        articles_with_content = self.add_article_content(articles)
        articles_json = self.articles_to_json(articles_with_content)
        df = self.json_to_csv(articles_json, csv_file_path)
        
        logger.info(f"Search completed. Found {len(df)} articles.")
        return df
