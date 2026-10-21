"""
Download authentic Indian car accident pictures from Wikimedia Commons and Flickr.
Saves verified images and a comprehensive METADATA.csv.
"""

import os
import sys
import time
import re
import csv
import io
import requests
from pathlib import Path
from PIL import Image

OUT_DIR = Path(r"c:\Hackathon Confluentia\Aequor\test_photos\indian_accidents")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UA = {'User-Agent': 'AequorIndianAccidentCollector/1.0 (academic vehicle damage research; contact@confluentia.edu)'}

WIKI_QUERIES = [
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
    'filetype:bitmap "accident" Tamil Nadu car',
    'filetype:bitmap "accident" Kolkata car',
    'filetype:bitmap "accident" Hyderabad car',
    'filetype:bitmap "accident" expressway India',
    'filetype:bitmap "car" "accident" highway India',
    'filetype:bitmap "car crash" Yamuna expressway',
    'filetype:bitmap "car crash" Mumbai Pune expressway',
    'filetype:bitmap "accident" Tata car',
    'filetype:bitmap "accident" Mahindra car'
]

FLICKR_TAG_SETS = [
    'car,accident,india',
    'car,crash,india',
    'road,accident,india',
    'traffic,accident,india',
    'damaged,car,india',
    'vehicle,damage,india',
    'delhi,car,accident',
    'mumbai,car,accident',
    'bangalore,car,accident',
    'kerala,car,accident',
    'kolkata,car,accident',
    'maruti,accident',
    'accident,india,highway',
    'vehicle,accident,india',
    'expressway,accident,india',
    'auto,accident,india',
    'car,collision,india'
]

EXCLUDE_KEYWORDS = [
    'medal', 'coin', 'stamp', 'treatise', 'memoir', 'chapel', 'prajnadas',
    'portrait', 'document', 'statue', 'hall', 'church', 'history', 'law',
    'surveys', 'evidence', 'charter', 'building'
]


def sanitize_filename(title: str, max_len: int = 40) -> str:
    clean = re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_').lower()
    return clean[:max_len] if clean else 'accident_img'


def collect_candidate_urls():
    candidates = []
    seen_urls = set()

    print("[1/3] Querying Wikimedia Commons for Indian car accident pictures...")
    for q in WIKI_QUERIES:
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
            f"&gsrnamespace=6&gsrsearch={requests.utils.quote(q)}"
            "&gsrlimit=50&prop=imageinfo&iiprop=url|mime|size&format=json"
        )
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
                        candidates.append({
                            'source': 'wikimedia',
                            'query': q,
                            'title': title,
                            'url': img_url
                        })
        except Exception as e:
            print(f"  Warning on query '{q}': {e}")
        time.sleep(0.5)

    print(f"  -> Discovered {len(candidates)} candidate images from Wikimedia Commons")

    print("[2/3] Querying Flickr public feeds for Indian car accident pictures...")
    flickr_count = 0
    for tags in FLICKR_TAG_SETS:
        url = f"https://api.flickr.com/services/feeds/photos_public.gne?tags={tags}&tagmode=all&format=json&nojsoncallback=1"
        try:
            r = requests.get(url, headers=UA, timeout=15)
            if r.status_code == 200:
                items = r.json().get('items', [])
                for it in items:
                    title = it.get('title', '')
                    title_lower = title.lower()
                    if any(ex in title_lower for ex in EXCLUDE_KEYWORDS):
                        continue
                    thumb_url = it.get('media', {}).get('m', '')
                    if not thumb_url:
                        continue
                    # Upgrade to higher resolution image
                    hires_url = thumb_url.replace('_m.jpg', '_b.jpg')
                    if hires_url not in seen_urls:
                        seen_urls.add(hires_url)
                        candidates.append({
                            'source': 'flickr',
                            'query': tags,
                            'title': title or tags,
                            'url': hires_url
                        })
                        flickr_count += 1
        except Exception as e:
            print(f"  Warning on Flickr tags '{tags}': {e}")
        time.sleep(0.5)

    print(f"  -> Discovered {flickr_count} candidate images from Flickr")
    print(f"Total unique candidates to download: {len(candidates)}")
    return candidates


def download_and_verify(candidates):
    print("\n[3/3] Downloading, validating and saving images...")
    saved_records = []
    idx = 1

    for cand in candidates:
        url = cand['url']
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code != 200:
                continue

            content = r.content
            if len(content) < 15000:  # Skip files smaller than 15KB
                continue

            # Verify with PIL
            try:
                img = Image.open(io.BytesIO(content))
                img.verify()
                img = Image.open(io.BytesIO(content))  # Reopen after verify
                width, height = img.size
                img_format = (img.format or 'JPEG').lower()
            except Exception:
                continue

            # Minimum size check (filter out tiny icons or banners)
            if width < 300 or height < 200:
                continue

            # Decide extension
            ext = '.jpg'
            if img_format in ['png', 'webp']:
                ext = f'.{img_format}'

            slug = sanitize_filename(cand['title'])
            filename = f"in_acc_{idx:03d}_{slug}{ext}"
            filepath = OUT_DIR / filename

            with open(filepath, 'wb') as f:
                f.write(content)

            cand['id'] = f"IN-ACC-{idx:03d}"
            cand['filename'] = filename
            cand['width'] = width
            cand['height'] = height
            cand['size_kb'] = round(len(content) / 1024, 1)
            cand['format'] = img_format.upper()
            saved_records.append(cand)

            print(f"  [SAVED #{idx:03d}] {filename} ({width}x{height}, {cand['size_kb']} KB)")
            idx += 1

        except Exception as err:
            # Skip transient errors
            continue

    # Write METADATA.csv
    csv_path = OUT_DIR / "METADATA.csv"
    fieldnames = ['id', 'filename', 'source', 'query', 'title', 'width', 'height', 'size_kb', 'format', 'url']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for rec in saved_records:
            writer.writerow(rec)

    print(f"\n========================================================")
    print(f"SUCCESS: Downloaded {len(saved_records)} verified Indian car accident pictures!")
    print(f"Destination: {OUT_DIR}")
    print(f"Metadata index: {csv_path}")
    print(f"========================================================")
    return saved_records


if __name__ == '__main__':
    candidates = collect_candidate_urls()
    download_and_verify(candidates)
