import requests


url="https://api.bybit.com/v5/market/instruments-info"


params={
    "category":"linear"
}


r=requests.get(
    url,
    params=params,
    timeout=20
)


data=r.json()


symbols=[]


for item in data["result"]["list"]:

    symbol=item["symbol"]


    if (
        "XAU" in symbol
        or
        "GOLD" in symbol
        or
        "PAXG" in symbol
    ):

        symbols.append(symbol)



print(symbols)