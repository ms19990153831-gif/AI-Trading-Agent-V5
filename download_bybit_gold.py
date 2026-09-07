import requests
import pandas as pd

from datetime import datetime, timedelta
import time


# ====================
# 参数
# ====================


SYMBOL = "XAUTUSDT"

CATEGORY = "linear"

INTERVAL = "1"


OUTPUT = "data/bybit_gold_m1.csv"



# 一年

DAYS = 365



BASE_URL = (
    "https://api.bybit.com/v5/market/kline"
)



# ====================
# 下载函数
# ====================


def get_kline(
        start,
        end
):


    params={

        "category":CATEGORY,

        "symbol":SYMBOL,

        "interval":INTERVAL,

        "start":int(
            start.timestamp()*1000
        ),

        "end":int(
            end.timestamp()*1000
        ),

        "limit":1000

    }



    r=requests.get(

        BASE_URL,

        params=params

    )


    data=r.json()



    if data["retCode"]!=0:

        print(data)

        return []


    return data["result"]["list"]




# ====================
# 主程序
# ====================


end=datetime.utcnow()


start=end-timedelta(
    days=DAYS
)



all_data=[]


current=start



while current < end:


    next_time=current+timedelta(
        minutes=1000
    )


    print(
        "下载:",
        current,
        next_time
    )



    rows=get_kline(

        current,

        next_time

    )



    if rows:

        all_data.extend(
            rows
        )



    current=next_time


    time.sleep(
        0.2
    )




print(
    "数据数量:",
    len(all_data)
)



# ====================
# 转换
# ====================


df=pd.DataFrame(

    all_data,

    columns=[

        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover"

    ]

)



df["time"]=pd.to_datetime(

    df["time"].astype(
        int
    ),

    unit="ms"

)



df=df.sort_values(
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



df.to_csv(

    OUTPUT,

    index=False

)



print(
"完成:",
OUTPUT
)