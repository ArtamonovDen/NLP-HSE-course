import requests
import json
import time
import random
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import timedelta, date

def parse_article(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'lxml')
    article = soup.body.find_next(id='article')
    
    title = soup.body.find_next(id='article').h1.get_text()
    category = article.find_next('a').get_text()
    
    # Collect text
    # Some hacks applied: as a rule, the last <p> is the author. Besides, we need to remove \xa0(NO-BREAK SPACE) symbol
    text = [p.get_text() for p in article.find_all('p',attrs={'class': None})]
    if article.find_all('p',attrs={'class': None})[-1].i:
        text.pop() # remove the author
    text = '\n'.join(text)
    text = text.replace(u'\xa0', u' ')
    
    return {'id':url, 'title':title, 'category':category, 'text':text}

def get_articles_hrefs(url):
    '''Returns href of news for a day'''
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'lxml')
    
    prefix = 'https://nplus1.ru/'
    return [prefix + art.a['href'] for art in soup.find_all('article')]


def get_days(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield (start_date + timedelta(n)).strftime("%Y/%m/%d")


def collect_data(start_date, end_date, out_file='nplus1_news.json'):
    collection = []
    prefix = 'https://nplus1.ru/news/'
    for day in tqdm(get_days(start_date, end_date)):
        url = prefix + day
        day_articles = get_articles_hrefs(url)
        for article in day_articles:
            parsed_article = parse_article(article)
            collection.append(parsed_article)
            
    with open(out_file, 'w') as file:
        json.dump(collection, file, ensure_ascii=False)
    