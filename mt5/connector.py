"""MT5 terminal connection with simulation fallback."""

from __future__ import annotations

from config import MT5_LOGIN, MT5_PASSWORD, MT5_PATH, MT5_SERVER, SIMULATE


class MT5Connector:
    def __init__(self) -> None:
        self.connected = False

    def connect(self) -> bool:
        if SIMULATE:
            self.connected = False
            return False
        import MetaTrader5 as mt5

        kwargs: dict = {}
        if MT5_PATH:
            kwargs["path"] = MT5_PATH
        if MT5_LOGIN:
            kwargs["login"] = MT5_LOGIN
            kwargs["password"] = MT5_PASSWORD
            kwargs["server"] = MT5_SERVER
        self.connected = bool(mt5.initialize(**kwargs))
        return self.connected

    def shutdown(self) -> None:
        if not SIMULATE:
            try:
                import MetaTrader5 as mt5

                mt5.shutdown()
            except Exception:
                pass
        self.connected = False
