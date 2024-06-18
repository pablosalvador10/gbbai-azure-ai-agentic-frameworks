

def enhanced_pubmed_search(query, max_results, email):
    """
    Conducts an enhanced PubMed search for articles relevant to medical research, matching the specified query, and returns the top results.
    
    Args:
    - query (str): The search query, tailored for medical research.
    - max_results (int): The maximum number of results to return. Defaults to 10.
    - email (str): Your email address for PubMed API usage, specific to medical research.
    
    Returns:
    - List[Dict[str, Any]]: A list of article summaries relevant to medical research in dictionary format.
    """
    from src.tools.pubmed import PubMedScraper
    EMAIL="pablosalvadorlopez11@gmail.com"
    scraper = PubMedScraper(email=email)
    articles = scraper.search_by_query(query, max_results)
    return articles.to_dict(orient='records')


