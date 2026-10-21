import requests

UA = {'User-Agent': 'AequorResearch/1.0 (academic vehicle damage research)'}

queries = [
    'filetype:bitmap "car accident" India',
    'filetype:bitmap "traffic accident" India',
    'filetype:bitmap "road accident" India',
    'filetype:bitmap "car crash" India',
    'filetype:bitmap "damaged car" India',
    'filetype:bitmap "vehicle accident" India',
    'filetype:bitmap "accident" India car',
    'filetype:bitmap "accident" Delhi car',
    'filetype:bitmap "accident" Mumbai car',
    'filetype:bitmap "accident" Bangalore car'
]

all_titles = set()
for q in queries:
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch={requests.utils.quote(q)}&gsrlimit=50&prop=imageinfo&iiprop=url|mime|size&format=json"
    res = requests.get(url, headers=UA).json()
    pages = res.get('query', {}).get('pages', {})
    for pid, pdata in pages.items():
        title = pdata.get('title', '')
        ii = pdata.get('imageinfo', [{}])[0]
        mime = ii.get('mime', '')
        url_img = ii.get('url', '')
        if mime.startswith('image/'):
            all_titles.add((title, url_img))
    print(f"Query '{q}': total unique image files so far: {len(all_titles)}")
