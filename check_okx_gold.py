import requests


url="https://www.okx.com/api/v5/public/instruments"


categories=[
    "SPOT",
    "SWAP",
    "FUTURES"
]


for category in categories:


    print("================")
    print(category)


    params={
        "instType":category
    }


    r=requests.get(
        url,
        params=params,
        timeout=20
    )


    data=r.json()


    if data.get("code")!="0":
        print(data)
        continue



    result=[]


    for item in data["data"]:

        inst=item["instId"]


        if (
            "XAU" in inst
            or
            "XAUT" in inst
            or
            "GOLD" in inst
            or
            "PAXG" in inst
        ):

            result.append(inst)



    print(result)