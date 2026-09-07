import websocket
import json
import ssl


def on_open(ws):

    print("OKX WebSocket连接成功")


    subscribe = {

        "op": "subscribe",

        "args": [

            {
                "channel": "tickers",
                "instId": "BTC-USDT-SWAP"
            }

        ]

    }


    ws.send(
        json.dumps(subscribe)
    )


def on_message(ws, message):

    print("收到行情:")

    print(message[:500])



def on_error(ws, error):

    print("错误:")

    print(error)



def on_close(ws, code, msg):

    print("关闭:")

    print(code, msg)



print("启动OKX WebSocket测试")


ws = websocket.WebSocketApp(

    "wss://ws.okx.com:443/ws/v5/public",

    header=[

        "User-Agent: Mozilla/5.0"

    ],

    on_open=on_open,

    on_message=on_message,

    on_error=on_error,

    on_close=on_close

)



ws.run_forever(

    ping_interval=15,

    ping_timeout=10,

    sslopt={

        "cert_reqs": ssl.CERT_NONE

    }

)