import pandas as pd


INPUT="data/mt5_xauusd_tick.csv"

OUTPUT="data/mt5_xauusd_tick_m1.csv"



chunks=[]


print("开始读取Tick数据...")


for i,chunk in enumerate(
    pd.read_csv(
        INPUT,
        chunksize=500000,
        encoding="ascii",
        on_bad_lines="skip"
    )
):

    print(
        "处理区块:",
        i
    )


    chunk["time"]=pd.to_datetime(
        chunk["time"],
        errors="coerce"
    )


    chunk=chunk.dropna(
        subset=["time"]
    )


    chunk=chunk.set_index(
        "time"
    )


    m1=chunk.resample(
        "1min"
    ).last()


    m1=m1.dropna()


    chunks.append(m1)



print("合并数据...")


df=pd.concat(
    chunks
)



df=df.sort_index()



df=df[
[
"bid",
"ask"
]
]



df=df.reset_index()



df.to_csv(
    OUTPUT,
    index=False
)



print("================")

print(
"转换完成"
)


print(
"分钟数量:",
len(df)
)


print(
"输出:",
OUTPUT
)