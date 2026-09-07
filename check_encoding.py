import chardet


filename = "data/mt5_xauusd_tick.csv"


with open(filename, "rb") as f:

    raw = f.read(200000)


result = chardet.detect(raw)


print(result)