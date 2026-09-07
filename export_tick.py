import MetaTrader5 as mt5
import pandas as pd

from datetime import datetime, timedelta
import os



SYMBOL="XAUUSD"

OUTPUT="data/mt5_xauusd_tick.csv"



if not mt5.initialize():

    print(
        "MT5连接失败",
        mt5.last_error()
    )

    quit()



print("MT5连接成功")



end=datetime.now()

start=end-timedelta(days=365)



all_data=[]


current=start



while current < end:


    next_day=current+timedelta(days=1)



    print(
        "下载:",
        current,
        "到",
        next_day
    )



    ticks=mt5.copy_ticks_range(

        SYMBOL,

        current,

        next_day,

        mt5.COPY_TICKS_ALL

    )



    if ticks is not None and len(ticks)>0:


        df=pd.DataFrame(
            ticks
        )


        all_data.append(df)


        print(
            "数量:",
            len(df)
        )


    else:

        print(
            "无数据"
        )



    current=next_day



# 合并


if len(all_data)==0:

    print(
        "没有获取任何Tick"
    )

    mt5.shutdown()

    quit()



df=pd.concat(
    all_data,
    ignore_index=True
)



df["time"]=pd.to_datetime(

    df["time"],

    unit="s"

)



# 保留套利需要字段

df=df[

[
"time",
"bid",
"ask",
"last",
"volume"
]

]



df.to_csv(

    OUTPUT,

    index=False

)



print("===================")

print(
"Tick导出完成"
)


print(
"总数量:",
len(df)
)


print(
"文件:",
OUTPUT
)



mt5.shutdown()