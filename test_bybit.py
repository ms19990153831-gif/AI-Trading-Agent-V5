import requests


url="https://api.bybit.com/v5/market/time"


r=requests.get(
    url,
    timeout=10
)


print(
    r.text
)