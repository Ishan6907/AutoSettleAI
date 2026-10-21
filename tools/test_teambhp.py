import requests
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# The famous thread on Team-BHP is "Accidents in India | Pictures & videos"
# Thread id: 10928 or similar
url = "https://www.team-bhp.com/forum/road-safety/10928-accidents-india-pictures-videos.html"
try:
    resp = requests.get(url, headers=headers, timeout=10)
    print("Team-BHP Status:", resp.status_code)
    # Search for image attachments or img src
    imgs = re.findall(r'src="(https?://[^\s"]+?\.(?:jpg|jpeg|png))"', resp.text, re.IGNORECASE)
    print(f"Found {len(imgs)} images on page 1")
    for img in imgs[:5]:
        print(img)
except Exception as e:
    print("Error:", e)
