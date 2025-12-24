"""Fetch Polymarket events matching stock price prediction pattern."""

import json
import logging
import re
from datetime import UTC, datetime

import requests
from domain.models import EventMetadata, MarketMetadata
from domain.types import Symbol, is_symbol

# Weekly pattern: "Will Amazon (AMZN) finish week of December 29 above___?"
WEEKLY_EVENT_PATTERN = re.compile(r"^Will .+ \(([A-Z]+)\) finish week of .+ above___\?$")
WEEKLY_MARKET_PATTERN = re.compile(r"^Will .+ \([A-Z]+\) finish week of .+ above \$?([\d.]+)\?$")

# Monthly pattern: "Will NVIDIA (NVDA) close above ___ end of January?"
MONTHLY_EVENT_PATTERN = re.compile(r"^Will .+ \(([A-Z]+)\) close above ___ end of \w+\?$")
MONTHLY_MARKET_PATTERN = re.compile(r"^Will .+ \([A-Z]+\) close above \$?([\d.]+) end of \w+\?$")

GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
BATCH_SIZE = 500

logger = logging.getLogger(__name__)


def fetch_stock_events() -> list[EventMetadata]:
    """Fetch events from Polymarket Gamma API and filter for stock price questions."""
    session = requests.Session()
    end_of_year = f"{datetime.now(UTC).year}-12-31T23:59:59Z"

    params = {
        "closed": "false",
        "end_date_max": end_of_year,
        "limit": BATCH_SIZE,
    }

    matching_events: list[EventMetadata] = []
    offset = 0

    while True:
        params["offset"] = offset
        response = session.get(GAMMA_API_URL, params=params, timeout=30)
        response.raise_for_status()

        events = response.json()
        if not events:
            break

        for event in events:
            event_question = event.get("title", "")
            if WEEKLY_EVENT_PATTERN.match(event_question) or MONTHLY_EVENT_PATTERN.match(event_question):
                markets = []
                all_markets = event.get("markets", [])

                for market in all_markets:
                    market_info = _get_useful_market_info(market)
                    if market_info:
                        markets.append(market_info)

                symbol = _parse_symbol(event_question)
                if not symbol:
                    continue

                end_date_str = event.get("endDate")
                end_date = datetime.fromisoformat(end_date_str) if end_date_str else None

                matching_events.append(
                    EventMetadata(
                        symbol=symbol,
                        question=event_question,
                        question_id=event.get("id"),
                        end_date=end_date,
                        markets=markets,
                    )
                )

        offset += len(events)
        if len(events) < BATCH_SIZE:
            break

    return matching_events


def _get_useful_market_info(market: dict) -> MarketMetadata | None:
    """Extract essential market metadata from a Gamma API market dict."""
    tokens = json.loads(market.get("clobTokenIds", "[]"))
    prices = json.loads(market.get("outcomePrices", "[]"))
    market_question = market.get("question")
    if not market_question:
        return None

    strike_price = _parse_strike_price(market_question)
    if not strike_price:
        return None

    return MarketMetadata(
        question=market_question,
        question_id=market.get("questionID"),
        strike_price=strike_price,
        yes_token_id=tokens[0] if tokens else None,
        yes_price=prices[0] if prices else None,
        no_token_id=tokens[1] if len(tokens) > 1 else None,
        no_price=prices[1] if len(prices) > 1 else None,
        best_bid=market.get("bestBid"),
        best_ask=market.get("bestAsk"),
        tick_size=market.get("orderPriceMinTickSize"),
        neg_risk=market.get("negRisk"),
        fee_rate_bps=None,
    )


def _parse_symbol(event_question: str) -> Symbol | None:
    """Extract stock symbol from event title."""
    match = WEEKLY_EVENT_PATTERN.match(event_question)
    if not match:
        match = MONTHLY_EVENT_PATTERN.match(event_question)

    if match:
        symbol = match.group(1)
        if is_symbol(symbol):
            return symbol
        logger.warning("Unknown symbol parsed: %s", symbol)
        return None

    logger.warning("Could not parse symbol from question: %s", event_question)
    return None


def _parse_strike_price(market_question: str) -> float | None:
    """Extract strike price from market question."""
    match = WEEKLY_MARKET_PATTERN.match(market_question)
    if not match:
        match = MONTHLY_MARKET_PATTERN.match(market_question)

    if match:
        return float(match.group(1))

    logger.warning("Could not parse strike price from question: %s", market_question)
    return None
