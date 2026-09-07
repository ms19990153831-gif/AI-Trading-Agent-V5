import requests


url = "https://www.okx.com/api/v5/public/instruments"


params = {
    "instType": "SWAP"
}


try:

    r = requests.get(
        url,
        params=params,
        timeout=10
    )


    print("HTTP状态:")
    print(r.status_code)


    print("返回:")
    print(r.text[:500])


except Exception as e:

    print("错误:")
    print(e)