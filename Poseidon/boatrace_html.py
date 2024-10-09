import requests
from bs4 import BeautifulSoup

url = 'https://poseidon-boatrace.net/'  # 対象のURL
headers = {'User-Agent': 'Mozilla/5.0'}  # ヘッダー情報を追加

response = requests.get(url, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    # ここでsoupオブジェクトを使ってHTMLを解析します
    print(soup.prettify())
else:
    print(f'Error: {response.status_code}')
