#!/usr/bin/env python3
"""
Dump raw WebSocket messages from the options feed.

Uses the same symbol fetching logic as production OptionsQuoteListener.
"""

import asyncio

from domain.secrets import load_required_secret
from domain.types import SYMBOLS
from dotenv import load_dotenv
from listener.options_listener import fetch_options_symbols
from massive import WebSocketClient
from massive.websocket.models import EquityAgg, Feed, Market, WebSocketMessage

load_dotenv()


async def handle_message(messages: list[WebSocketMessage]) -> None:
    for msg in messages:
        if isinstance(msg, EquityAgg):
            print(f"{msg.symbol} | close={msg.close} vol={msg.volume} vwap={msg.vwap}")


async def main() -> None:
    api_key = load_required_secret("MASSIVE_API_KEY")

    print(f"Fetching options for: {SYMBOLS}")
    symbols = fetch_options_symbols(api_key, SYMBOLS)
    print(f"Subscribing to {len(symbols)} symbols...")

    subs = [f"A.{s}" for s in symbols]

    ws = WebSocketClient(
        api_key=api_key,
        feed=Feed.Delayed,
        market=Market.Options,
        subscriptions=subs,
    )
    await ws.connect(processor=handle_message)


if __name__ == "__main__":
    asyncio.run(main())
