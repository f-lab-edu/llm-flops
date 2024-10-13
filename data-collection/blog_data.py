import re
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options
import time

# Set up Firefox options
firefox_options = Options()
firefox_options.add_argument('--headless')  # Run in headless mode (no browser UI)
firefox_options.add_argument('--disable-gpu')
firefox_options.add_argument('--no-sandbox')

# Set up the WebDriver
service = Service(GeckoDriverManager().install())

class WebsiteDataCrawler:
    def __init__(self):
        self.driver = webdriver.Firefox(service=service, options=firefox_options)
        pass
    
    def get_all_hrefs(self, url:str):
        # URL에 대한 GET 요청 보내기
        response = requests.get(url)
        
        # BeautifulSoup을 사용하여 HTML 콘텐츠 파싱
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 모든 <a> 태그 찾기
        a_tags = soup.find_all('a')
     
        # 각 <a> 태그에서 href 속성 추출
        hrefs = [a.get('href') for a in a_tags if a.get('href') is not None]
        
        return hrefs
    
    def get_openai_suburls(self, n_recent=30):
        url = f'https://openai.com/news/?limit={n_recent}'
        openai_url_list = list()
        try:
            self.driver.get(url)

            # Wait for JavaScript to load (you may need to adjust the sleep duration or use explicit waits)
            time.sleep(2)  # Time to allow JavaScript to execute

            # Find all <a> tags
            links = self.driver.find_elements(By.TAG_NAME, 'a')
            for link in links:
                link_dir = link.get_attribute('href')
                if '/index/' in link_dir:
                    openai_url_list.append(link_dir)
        finally:
            # Quit the driver
            self.driver.quit()

        return list(set(openai_url_list))
    
    def get_anthropic_suburls(self):
        # anthropic 웹사이트 서브 URL 가져오기
        url = 'https://www.anthropic.com/news'
        site_href_list = self.get_all_hrefs(url)
        news_hrefs = [url + href.replace('/news', '') for href in site_href_list if '/news' in href and href != '/news']
        return list(set(news_hrefs))
        
    def get_ncsoft_suburls(self):
        # NCSoft 웹사이트 서브 URL 가져오기
        url = 'https://ncsoft.github.io/ncresearch/blogs/'
        site_href_list = self.get_all_hrefs(url)
        pattern = re.compile(r"^/ncresearch/[a-fA-F0-9]{40}$")
        filtered_paths = [url.replace('/ncresearch/blogs/', '') + path for path in site_href_list if pattern.match(path)]
        return filtered_paths
    
    def get_naver_suburls(self):
        # 네이버 웹사이트 서브 URL 가져오기
        url = 'https://clova.ai/tech-blog'
        site_href_list = list(set(self.get_all_hrefs(url)))
        filtered_path = [url.replace('/tech-blog', '') + path for path in site_href_list if ('/tech-blog' in path) and ('/tag/' not in path) and (path != '/tech-blog')]
        return filtered_path
    
    def get_all_docs(self):
        # 모든 문서 서브 URL을 통해 문서 목록 가져오기
        openai = self.get_openai_suburls()
        anthropic = self.get_anthropic_suburls()
        ncsoft = self.get_ncsoft_suburls()
        naver = self.get_naver_suburls()


        total_list_suburl = openai + anthropic + ncsoft + naver

        loader = WebBaseLoader(total_list_suburl)
        docs = loader.load()
        return docs
    

if __name__=='__main__':
    website_data = WebsiteDataCrawler()
    site_list = sorted(website_data.get_openai_suburls())
    for site in site_list:
        print(site)
    