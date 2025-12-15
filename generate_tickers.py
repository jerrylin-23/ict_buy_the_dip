import json
import urllib.request
import ssl

# Bypass SSL errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def get_json(url):
    try:
        with urllib.request.urlopen(url, context=ctx) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

# Custom Retail/High-Beta List (Excluding IWM)
custom = [
    "GME", "AMC", "PLTR", "COIN", "HOOD", "DKNG", "ROKU", "SHOP", "SQ", "TSLA",
    "NVDA", "AMD", "MARA", "RIOT", "MSTR", "CVNA", "UPST", "AFRM", "SOFI",
    "SPY", "QQQ", "DIA", "TLT", "UVXY", "VXX"
]

print("Fetching SP500 constituents...")
sp500_data = get_json("https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/sp500/components.json")
sp500 = [x['symbol'] for x in sp500_data]

print("Fetching Nasdaq 100 constituents...")
nasdaq_data = get_json("https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq100/components.json")
nasdaq = [x['symbol'] for x in nasdaq_data]

# Combine (excluding IWM/Russell components)
full_list = sorted(list(set(custom + sp500 + nasdaq)))

output_path = 'src/tickers.txt'
with open(output_path, 'w') as f:
    for ticker in full_list:
        f.write(f"{ticker}\n")

print(f"✅ Generated {len(full_list)} tickers in {output_path}")
print(f"❌ Excluded IWM and Russell 2000 components.")
