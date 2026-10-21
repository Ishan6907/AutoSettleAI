import requests
import json
import re
from pathlib import Path
from PIL import Image
import io

UA = {'User-Agent': 'AequorIndianAccidentCollector/1.0 (academic vehicle damage research; contact@confluentia.edu)'}

wiki_queries = [
    'filetype:bitmap "car accident" India',
    'filetype:bitmap "road accident" India',
    'filetype:bitmap "traffic accident" India',
    'filetype:bitmap "car crash" India',
    'filetype:bitmap "damaged car" India',
    'filetype:bitmap "damaged vehicle" India',
    'filetype:bitmap "overturned" India car',
    'filetype:bitmap "vehicle collision" India',
    'filetype:bitmap "motor vehicle accident" India',
    'filetype:bitmap Maruti accident',
    'filetype:bitmap Swift accident',
    'filetype:bitmap "accident" Delhi car',
    'filetype:bitmap "accident" Mumbai car',
    'filetype:bitmap "accident" Bangalore car',
    'filetype:bitmap "accident" Kerala car',
    'filetype:bitmap "car" "accident" highway India'
]

flickr_tags = [
    'car,accident,india',
    'car,crash,india',
    'road,accident,india',
    'traffic,accident,india',
    'damaged,car,india',
    'delhi,car,accident',
    'mumbai,car,accident',
    'bangalore,car,accident',
    'kerala,car,accident',
    'maruti,accident',
    'accident,india,highway',
    'vehicle,accident,india'
]

# Exclude obvious non-damage keywords (e.g., medals, coins, books, stamps)
EXCLUDE_KEYWORDS = ['medal', 'coin', 'stamp', 'treatise', 'memoir', 'chapel', 'prajnadas', 'portrait', 'document', 'statue']

found_images = []
seen_urls = set()

# 1. Wikimedia
print("Scanning Wikimedia Commons...")
for q in wiki_queries:
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch={requests.utils.quote(q)}&gsrlimit=50&prop=imageinfo&iiprop=url|mime|size&format=json"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code == 200:
            pages = r.json().get('query', {}).get('pages', {})
            for pid, pdata in pages.items():
                title = pdata.get('title', '')
                ii = pdata.get('imageinfo', [{}])[0]
                mime = ii.get('mime', '')
                img_url = ii.get('url', '')
                
                title_lower = title.lower()
                if any(ex in title_lower for ex in EXCLUDE_KEYWORDS):
                    continue
                if not (mime.startswith('image/') and not mime.endswith('svg+xml')):
                    continue
                if img_url and img_url not in seen_urls:
                    seen_urls.add(img_url)
                    found_images.append({
                        'source': 'wikimedia',
                        'query': q,
                        'title': title,
                        'url': img_url
                    })
    except Exception as e:
        print(f"Error on {q}: {e}")

print(f"Wikimedia gathered: {len(found_images)} candidate images")

# 2. Flickr
print("\nScanning Flickr feeds...")
flickr_count = 0
for tags in flickr_tags:
    url = f"https://api.flickr.com/services/feeds/photos_public.gne?tags={tags}&tagmode=all&format=json&nojsoncallback=1"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            items = r.json().get('items', [])
            for it in items:
                title = it.get('title', '')
                title_lower = title.lower()
                if any(ex in title_lower for ex in EXCLUDE_KEYWORDS):
                    continue
                # Media m is thumbnail, replace _m.jpg with _b.jpg (1024px)
                thumb_url = it.get('media', {}).get('m', '')
                if not thumb_url:
                    continue
                hires_url = thumb_url.replace('_m.jpg', '_b.jpg')
                if hires_url not in seen_urls:
                    seen_urls.add(hires_url)
                    found_images.append({
                        'source': 'flickr',
                        'query': tags,
                        'title': title or tags,
                        'url': hires_url
                    })
                    flickr_count += 1
    except Exception as e:
        print(f"Error on Flickr {tags}: {e}")

print(f"Flickr gathered: {flickr_count} candidate images")
print(f"TOTAL candidate images across all sources: {len(found_images)}")

for i, img in enumerate(found_images[:10]):
    print(f"[{i+1}] ({img['source']}) {img['title']} -> {img['url']}")
