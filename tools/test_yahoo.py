import requests
import re
from urllib.parse import unquote

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}

resp = requests.get('https://images.search.yahoo.com/search/images?p=indian+car+accident+pictures', headers=headers, timeout=10)
print('Yahoo status:', resp.status_code)
# Yahoo images usually has imgurl= or rurl=
img_urls = re.findall(r'imgurl=(http[^&]+)', resp.text)
print(f'Found {len(img_urls)} imgurl matches in Yahoo')
for u in img_urls[:5]:
    print('Unquoted:', unquote(u))
