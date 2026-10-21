import requests
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

resp = requests.get('https://www.google.com/search?q=indian+car+accident+pictures&udm=2', headers=headers, timeout=10)
print('Google status:', resp.status_code)

# Check for https://encrypted-tbn0.gstatic.com/images?q=tbn:...
tbn_urls = re.findall(r'https://encrypted-tbn0\.gstatic\.com/images\?q=tbn:[^"\'&]+', resp.text)
print('Encrypted tbn URLs:', len(tbn_urls))

# Also search for original image links:
# In Google Images (udm=2), original URLs are often in data-src or in nested JSON
orig_urls = re.findall(r'(https?://[^\s"\'<>]+?\.(?:jpg|jpeg|png))', resp.text, re.IGNORECASE)
print('Any jpg/jpeg/png URLs in HTML:', len(orig_urls))
for u in list(set(orig_urls))[:10]:
    if 'google' not in u and 'gstatic' not in u:
        print('External image:', u)
