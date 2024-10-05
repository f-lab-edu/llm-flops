import re
import requests
from bs4 import BeautifulSoup

class WebsiteDataCrawler:
    def __init__(self):
        pass
    
    def get_all_hrefs(self, url:str):
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all <a> tags
        a_tags = soup.find_all('a')
     
        # Extract href attributes from each <a> tag
        hrefs = [a.get('href') for a in a_tags if a.get('href') is not None]
        
        return hrefs
    

    def get_anthropic_suburls(self):
        url = 'https://www.anthropic.com/news'
        site_href_list = self.get_all_hrefs(url)
        news_hrefs = [url + href.replace('/news', '') for href in site_href_list if '/news' in href and href != '/news']
        return list(set(news_hrefs))
        
    def get_ncsoft_suburls(self):
        url = 'https://ncsoft.github.io/ncresearch/blogs/'
        site_href_list = self.get_all_hrefs(url)
        pattern = re.compile(r"^/ncresearch/[a-fA-F0-9]{40}$")
        filtered_paths = [url.replace('/ncresearch/blogs/', '') + path for path in site_href_list if pattern.match(path)]
        return filtered_paths
    
    def get_naver_suburls(self):
        url = 'https://clova.ai/tech-blog'
        site_href_list = list(set(self.get_all_hrefs(url)))
        filtered_path = [url.replace('/tech-blog', '') + path for path in site_href_list if ('/tech-blog' in path) and ('/tag/' not in path) and (path != '/tech-blog')]
        return filtered_path
    


if __name__=='__main__':
    website_data = WebsiteDataCrawler()
    site_list = sorted(website_data.get_naver_suburls())
    for site in site_list:
        print(site)
    