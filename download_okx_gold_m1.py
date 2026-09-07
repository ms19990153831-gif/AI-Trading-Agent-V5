import requests
import pandas as pd
import time
import os

from datetime import datetime, timedelta


INST_ID = "XAU-USDT-SWAP"

BAR = "1m"

OUTPUT = "data/okx_xau_swap_m1.csv"

API = "https://www.okx.com/api/v5/market/history-candles"


DAYS = 30



def get_candles(after=None):

    params = {

        "instId": INST_ID,

        "bar": BAR,

        "limit": 100

    }


    if after:

        params["after"] = after



    for i in range(5):

        try:

            r = requests.get(

                API,

                params=params,

                timeout=20,

                headers={
                    "User-Agent":"Mozilla/5.0"
                }

            )


            data=r.json()


            if data["code"]=="0":

                return data["data"]


            else:

                print(data)


        except Exception as e:

            print(
                "请求失败:",
                e
            )


        time.sleep(3)


    return []



# =====================

print(
    "开始下载",
    INST_ID
)


all_rows=[]


cursor=None


start_limit=datetime.now()-timedelta(days=DAYS)



while True:


    rows=get_candles(cursor)


    if not rows:

        print(
            "没有更多数据"
        )

        break



    print(
        "本次:",
        len(rows)
    )



    all_rows.extend(rows)



    # OKX返回:
    # 最新在前
    # 最老在后

    cursor=rows[-1][0]



    oldest=datetime.fromtimestamp(

        int(cursor)/1000

    )


    print(
        "当前最早:",
        oldest
    )



    if oldest < start_limit:

        print(
            "达到一年"
        )

        break



    time.sleep(0.5)




print(
    "总数量:",
    len(all_rows)
)



# =====================

# 转换

# =====================


df=pd.DataFrame(

    all_rows,

    columns=[

        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volCcy",
        "volCcyQuote",
        "confirm"

    ]

)



df["time"]=pd.to_datetime(

    df["time"].astype(int),

    unit="ms"

)



for c in [

"open",
"high",
"low",
"close",
"volume"

]:

    df[c]=df[c].astype(float)



df=df.sort_values(
    "time"
)



df=df.drop_duplicates(
    "time"
)



df=df[

[
"time",
"open",
"high",
"low",
"close",
"volume"
]

]


os.makedirs(
    "data",
    exist_ok=True
)



df.to_csv(

    OUTPUT,

    index=False

)



print("================")

print(
"完成"
)

print(
"文件:",
OUTPUT
)

print(
"数量:",
len(df)
)