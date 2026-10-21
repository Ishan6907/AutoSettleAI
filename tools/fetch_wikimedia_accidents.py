import requests
import time

UA = {'User-Agent': 'AequorResearch/1.0 (academic vehicle damage research; contact@confluentia.edu)'}

queries = [
    'filetype:bitmap "car accident" India',
    'filetype:bitmap "road accident" India',
    'filetype:bitmap "traffic accident" India',
    'filetype:bitmap "car crash" India',
    'filetype:bitmap "damaged car" India',
    'filetype:bitmap "damaged vehicle" India',
    'filetype:bitmap "overturned car" India',
    'filetype:bitmap "vehicle collision" India',
    'filetype:bitmap "motor vehicle accident" India',
    'filetype:bitmap Maruti accident',
    'filetype:bitmap Swift accident',
    'filetype:bitmap "car" "accident" Delhi',
    'filetype:bitmap "car" "accident" Mumbai',
    'filetype:bitmap "car" "accident" Bangalore',
    'filetype:bitmap "car" "accident" Kerala',
    'filetype:bitmap "car" "accident" Tamil Nadu',
    'filetype:bitmap "car" "accident" highway India'
]

results = []
seen = set()

for q in queries:
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch={requests.utils.quote(q)}&gsrlimit=50&prop=imageinfo&iiprop=url|mime|size&format=json"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code == 200:
            data = r.json()
            pages = data.get('query', {}).get('pages', {})
            for pid, pdata in pages.items():
                title = pdata.get('title', '')
                ii = pdata.get('imageinfo', [{}])[0]
                mime = ii.get('mime', '')
                img_url = ii.get('url', '')
                if mime.startswith('image/') and not mime.endswith('svg+xml'):
                    if img_url and img_url not in seen:
                        seen.add(img_url)
                        results.append({'title': title, 'url': img_url, 'query': q})
        print(f"Query '{q}': total found so far = {len(results)}")
    except Exception as e:
        print(f"Error on {q}: {e}")
    time.sleep(1.5)

print(f"\nTOTAL unique Wikimedia images found: {len(results)}")
for r in results[:10]:
    print(r['title'], r['url'])
