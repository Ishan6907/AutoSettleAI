import requests

url = "https://api.flickr.com/services/feeds/photos_public.gne?tags=car,accident,india&tagmode=all&format=json&nojsoncallback=1"
try:
    r = requests.get(url, timeout=10)
    data = r.json()
    items = data.get('items', [])
    print(f"Flickr public feed returned {len(items)} items")
    for it in items[:3]:
        print(it.get('title'), it.get('media', {}).get('m'))
except Exception as e:
    print("Flickr error:", e)
