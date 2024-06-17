import requests
import xml.etree.ElementTree as ET
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from Bio import Entrez
import time

class PubMedScraper:
    def __init__(self, email: str):
        self.email = email
        Entrez.email = email
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def fetch_pubmed_ids(self, query: str, max_results: int = 10, field: str = None, sort: str = "relevance", 
                         datetype: str = None, mindate: str = None, maxdate: str = None, 
                         retstart: int = 0) -> List[str]:
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
        response.raise_for_status()
        data = response.json()
        return data.get('esearchresult', {}).get('idlist', [])

    def fetch_pubmed_articles(self, ids: List[str]) -> str:
        handle = Entrez.efetch("pubmed", id=",".join(ids), retmode="xml")
        return handle.read()

    def parse_article_details(self, xml_data: str) -> List[Dict[str, Any]]:
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
        return json.dumps(articles, indent=4)

    def fetch_pdf_content(self, url: str) -> Optional[str]:
        retries = 3
        for _ in range(retries):
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
                if e.response.status_code == 403:
                    time.sleep(2)
                    continue
                return f"Error fetching PDF content: {e}"
        return "Failed to fetch PDF content after retries."

    def add_article_content(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for article in articles:
            if 'pdf_link' in article and article['pdf_link'] != "No PDF link available":
                article['full_content'] = self.fetch_pdf_content(article['pdf_link'])
            else:
                article['full_content'] = "No full text link available"
        return articles

    def json_to_csv(self, json_data: str, csv_file_path: str = None) -> pd.DataFrame:
        articles = json.loads(json_data)
        columns = ['pmid', 'title', 'abstract', 'authors', 'year', 'volume', 'issue', 'journal', 'citation', 'link', 'pdf_link', 'full_content']
        df = pd.DataFrame(articles, columns=columns)
        if csv_file_path:
            df.to_csv(csv_file_path, index=False)
        return df
    
    def search_by_query(self, query: str, max_results: int = 10, field: Optional[str] = None, 
                        sort: str = "relevance", datetype: Optional[str] = None, 
                        mindate: Optional[str] = None, maxdate: Optional[str] = None, 
                        retstart: int = 0, csv_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Searches PubMed based on a query and exports the results to a pandas DataFrame. Optionally, 
        the results can also be exported to a CSV file.

        Parameters:
        - query (str): The search query.
        - max_results (int): Maximum number of results to fetch. Default is 10.
        - field (Optional[str]): The search field. Default is None.
        - sort (str): Sort order of the results. Default is "relevance".
        - datetype (Optional[str]): The type of date used for filtering. Default is None.
        - mindate (Optional[str]): The start date for filtering results. Default is None.
        - maxdate (Optional[str]): The end date for filtering results. Default is None.
        - retstart (int): The starting point for results retrieval. Default is 0.
        - csv_file_path (Optional[str]): Path to save the results as a CSV file. Default is None.

        Returns:
        pd.DataFrame: A DataFrame containing the search results.
        """
        # Fetch PubMed IDs based on the query
        pubmed_ids = self.fetch_pubmed_ids(query, max_results, field, sort, datetype, mindate, maxdate, retstart)
        
        # If no IDs were found, return an empty DataFrame
        if not pubmed_ids:
            print("No articles found for the given query.")
            return pd.DataFrame()
        
        # Fetch articles based on the PubMed IDs
        xml_data = self.fetch_pubmed_articles(pubmed_ids)
        
        # Parse the article details from the XML data
        articles = self.parse_article_details(xml_data)
        
        # Optionally, add full article content from PDFs
        articles_with_content = self.add_article_content(articles)
        
        # Convert the articles to JSON
        articles_json = self.articles_to_json(articles_with_content)
        
        # Convert the JSON to a DataFrame and optionally export to CSV
        df = self.json_to_csv(articles_json, csv_file_path)
        
        return df

