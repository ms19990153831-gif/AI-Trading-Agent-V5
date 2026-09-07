import MetaTrader5 as mt5
import pandas as pd

from datetime import datetime, timedelta


# =========================
# 参数
# =========================

SYMBOL = "XAUUSD"

TIMEFRAME = mt5.TIMEFRAME_M1


DAYS = 365


OUTPUT = "data/mt5_xauusd_m1.csv"



# =========================
# 连接MT5
# =========================

def connect_mt5():

    if not mt5.initialize():

        print(
            "MT5连接失败:",
            mt5.last_error()
        )

        quit()


    print(
        "MT5连接成功"
    )



# =========================
# 下载数据
# =========================

def download_data():


    end = datetime.now()


    start = (
        end
        -
        timedelta(days=DAYS)
    )


    print(
        "开始下载:"
    )

    print(start)

    print(end)



    rates = mt5.copy_rates_range(

        SYMBOL,

        TIMEFRAME,

        start,

        end

    )


    if rates is None:

        print(
            "没有获取到数据"
        )

        print(
            mt5.last_error()
        )

        quit()



    return rates



# =========================
# 保存CSV
# =========================

def save_csv(rates):


    df=pd.DataFrame(
        rates
    )


    df["time"] = pd.to_datetime(

        df["time"],

        unit="s"

    )


    df=df[

        [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread"
        ]

    ]



    df.to_csv(

        OUTPUT,

        index=False

    )



    print(
        "导出完成"
    )


    print(
        "数据数量:",
        len(df)
    )


    print(
        "文件:",
        OUTPUT
    )




# =========================
# 主程序
# =========================


if __name__=="__main__":


    connect_mt5()


    data=download_data()


    save_csv(data)


    mt5.shutdown()