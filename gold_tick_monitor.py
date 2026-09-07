import MetaTrader5 as mt5
import websocket
import json
import threading
import time
import pandas as pd
from datetime import datetime


# ==========================
# 参数
# ==========================

OKX_SYMBOL = "XAU-USDT-SWAP"

OUTPUT = "data/gold_tick_spread.csv"


# 最新价格缓存

mt5_tick = {
    "bid": None,
    "ask": None,
    "time": None
}


okx_tick = {
    "bid": None,
    "ask": None,
    "time": None
}


records=[]



# ==========================
# MT5连接
# ==========================


def init_mt5():

    if not mt5.initialize():

        print(
            "MT5连接失败",
            mt5.last_error()
        )

        quit()


    print("MT5连接成功")



# ==========================
# MT5 Tick线程
# ==========================


def mt5_worker():


    while True:


        tick = mt5.symbol_info_tick(
            "XAUUSD"
        )


        if tick:


            mt5_tick["bid"]=tick.bid

            mt5_tick["ask"]=tick.ask

            mt5_tick["time"]=time.time_ns()/1e6



        time.sleep(0.01)



# ==========================
# OKX websocket
# ==========================


def okx_message(ws,msg):


    data=json.loads(msg)



    if "data" not in data:

        return



    item=data["data"][0]



    bid=float(
        item["bidPx"]
    )


    ask=float(
        item["askPx"]
    )



    okx_tick["bid"]=bid

    okx_tick["ask"]=ask

    okx_tick["time"]=time.time_ns()/1e6





def okx_open(ws):


    sub={

        "op":"subscribe",

        "args":[

            {

            "channel":"tickers",

            "instId":OKX_SYMBOL

            }

        ]

    }


    ws.send(
        json.dumps(sub)
    )


    print(
        "OKX连接成功"
    )





def okx_worker():


    url="wss://ws.okx.com:8443/ws/v5/public"


    ws=websocket.WebSocketApp(

        url,

        on_open=okx_open,

        on_message=okx_message

    )


    ws.run_forever()



# ==========================
# 价差计算
# ==========================


def spread_monitor():


    while True:


        if (

            mt5_tick["bid"]
            and
            okx_tick["bid"]

        ):



            # 中间价

            mt5_mid=(

                mt5_tick["bid"]

                +

                mt5_tick["ask"]

            )/2



            okx_mid=(

                okx_tick["bid"]

                +

                okx_tick["ask"]

            )/2



            spread=okx_mid-mt5_mid



            now=datetime.now()



            records.append({

                "time":now,

                "mt5_bid":
                mt5_tick["bid"],

                "mt5_ask":
                mt5_tick["ask"],

                "okx_bid":
                okx_tick["bid"],

                "okx_ask":
                okx_tick["ask"],

                "spread":
                spread

            })



            print(

                now,

                "价差:",

                round(spread,3)

            )



        time.sleep(0.05)



# ==========================
# 保存
# ==========================


def save_loop():


    while True:


        time.sleep(60)



        if records:


            df=pd.DataFrame(
                records
            )


            df.to_csv(

                OUTPUT,

                index=False

            )


            print(
                "保存:",
                len(df)
            )



# ==========================
# 启动
# ==========================


init_mt5()



threads=[

threading.Thread(
target=mt5_worker
),

threading.Thread(
target=okx_worker
),

threading.Thread(
target=spread_monitor
),

threading.Thread(
target=save_loop
)

]



for t in threads:

    t.start()


