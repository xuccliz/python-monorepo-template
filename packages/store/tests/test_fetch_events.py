"""Tests for fetch_polymarket module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from store.fetch_events import (
    MONTHLY_EVENT_PATTERN,
    MONTHLY_MARKET_PATTERN,
    WEEKLY_EVENT_PATTERN,
    WEEKLY_MARKET_PATTERN,
    _get_useful_market_info,
    _parse_strike_price,
    _parse_symbol,
    fetch_stock_events,
)


class TestWeeklyEventPattern:
    """Tests for WEEKLY_EVENT_PATTERN regex."""

    def test_matches_valid_question(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29 above___?"
        match = WEEKLY_EVENT_PATTERN.match(question)
        assert match is not None
        assert match.group(1) == "AMZN"

    def test_matches_different_company(self) -> None:
        question = "Will Apple (AAPL) finish week of January 5 above___?"
        match = WEEKLY_EVENT_PATTERN.match(question)
        assert match is not None
        assert match.group(1) == "AAPL"

    def test_no_match_without_above_placeholder(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29?"
        match = WEEKLY_EVENT_PATTERN.match(question)
        assert match is None

    def test_no_match_different_format(self) -> None:
        question = "What will AMZN stock price be?"
        match = WEEKLY_EVENT_PATTERN.match(question)
        assert match is None


class TestMonthlyEventPattern:
    """Tests for MONTHLY_EVENT_PATTERN regex."""

    def test_matches_end_of_month(self) -> None:
        question = "Will NVIDIA (NVDA) close above ___ end of January?"
        match = MONTHLY_EVENT_PATTERN.match(question)
        assert match is not None
        assert match.group(1) == "NVDA"

    def test_matches_different_month(self) -> None:
        question = "Will Apple (AAPL) close above ___ end of February?"
        match = MONTHLY_EVENT_PATTERN.match(question)
        assert match is not None
        assert match.group(1) == "AAPL"

    def test_no_match_weekly_format(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29 above___?"
        match = MONTHLY_EVENT_PATTERN.match(question)
        assert match is None


class TestWeeklyMarketPattern:
    """Tests for WEEKLY_MARKET_PATTERN regex."""

    def test_matches_with_dollar_sign(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29 above $200?"
        match = WEEKLY_MARKET_PATTERN.match(question)
        assert match is not None
        assert match.group(1) == "200"

    def test_matches_without_dollar_sign(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29 above 200?"
        match = WEEKLY_MARKET_PATTERN.match(question)
        assert match is not None
        assert match.group(1) == "200"

    def test_matches_decimal_price(self) -> None:
        question = "Will Tesla (TSLA) finish week of January 5 above $250.50?"
        match = WEEKLY_MARKET_PATTERN.match(question)
        assert match is not None
        assert match.group(1) == "250.50"

    def test_no_match_placeholder(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29 above___?"
        match = WEEKLY_MARKET_PATTERN.match(question)
        assert match is None


class TestMonthlyMarketPattern:
    """Tests for MONTHLY_MARKET_PATTERN regex."""

    def test_matches_end_of_month(self) -> None:
        question = "Will NVIDIA (NVDA) close above $140 end of January?"
        match = MONTHLY_MARKET_PATTERN.match(question)
        assert match is not None
        assert match.group(1) == "140"

    def test_matches_decimal_price(self) -> None:
        question = "Will Apple (AAPL) close above $185.50 end of February?"
        match = MONTHLY_MARKET_PATTERN.match(question)
        assert match is not None
        assert match.group(1) == "185.50"

    def test_no_match_weekly_format(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29 above $200?"
        match = MONTHLY_MARKET_PATTERN.match(question)
        assert match is None

    def test_no_match_placeholder(self) -> None:
        question = "Will NVIDIA (NVDA) close above ___ end of January?"
        match = MONTHLY_MARKET_PATTERN.match(question)
        assert match is None


class TestParseSymbol:
    """Tests for _parse_symbol function."""

    def test_parses_valid_symbol(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29 above___?"
        result = _parse_symbol(question)
        assert result == "AMZN"

    def test_parses_apple_symbol(self) -> None:
        question = "Will Apple (AAPL) finish week of January 5 above___?"
        result = _parse_symbol(question)
        assert result == "AAPL"

    def test_parses_nvidia_symbol(self) -> None:
        question = "Will NVIDIA (NVDA) finish week of February 10 above___?"
        result = _parse_symbol(question)
        assert result == "NVDA"

    def test_parses_monthly_format(self) -> None:
        question = "Will NVIDIA (NVDA) close above ___ end of January?"
        result = _parse_symbol(question)
        assert result == "NVDA"

    def test_parses_monthly_format_different_month(self) -> None:
        question = "Will Apple (AAPL) close above ___ end of February?"
        result = _parse_symbol(question)
        assert result == "AAPL"

    def test_returns_none_for_unknown_symbol(self) -> None:
        question = "Will Unknown (UNKN) finish week of December 29 above___?"
        result = _parse_symbol(question)
        assert result is None

    def test_returns_none_for_invalid_format(self) -> None:
        question = "What is the price of AMZN?"
        result = _parse_symbol(question)
        assert result is None


class TestParseStrikePrice:
    """Tests for _parse_strike_price function."""

    def test_parses_integer_price(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29 above $200?"
        result = _parse_strike_price(question)
        assert result == 200.0

    def test_parses_decimal_price(self) -> None:
        question = "Will Tesla (TSLA) finish week of January 5 above $250.50?"
        result = _parse_strike_price(question)
        assert result == 250.50

    def test_parses_price_without_dollar_sign(self) -> None:
        question = "Will Apple (AAPL) finish week of February 10 above 175?"
        result = _parse_strike_price(question)
        assert result == 175.0

    def test_parses_monthly_format(self) -> None:
        question = "Will NVIDIA (NVDA) close above $140 end of January?"
        result = _parse_strike_price(question)
        assert result == 140.0

    def test_parses_monthly_format_decimal(self) -> None:
        question = "Will Apple (AAPL) close above $185.50 end of February?"
        result = _parse_strike_price(question)
        assert result == 185.50

    def test_returns_none_for_placeholder(self) -> None:
        question = "Will Amazon (AMZN) finish week of December 29 above___?"
        result = _parse_strike_price(question)
        assert result is None

    def test_returns_none_for_invalid_format(self) -> None:
        question = "What is the price of AMZN?"
        result = _parse_strike_price(question)
        assert result is None


class TestGetUsefulMarketInfo:
    """Tests for _get_useful_market_info function."""

    def test_extracts_market_metadata(self) -> None:
        market = {
            "question": "Will Amazon (AMZN) finish week of December 29 above $200?",
            "questionID": "0x123abc",
            "clobTokenIds": json.dumps(["token_yes", "token_no"]),
            "outcomePrices": json.dumps(["0.65", "0.35"]),
            "orderPriceMinTickSize": "0.01",
            "negRisk": True,
            "bestBid": "0.60",
            "bestAsk": "0.70",
        }
        result = _get_useful_market_info(market)
        assert result is not None
        assert result.question_id == "0x123abc"
        assert result.yes_token_id == "token_yes"
        assert result.no_token_id == "token_no"
        # Prices are stored as-is from the API (strings converted by pydantic)
        assert result.yes_price == 0.65
        assert result.no_price == 0.35
        assert result.tick_size == "0.01"
        assert result.neg_risk is True
        assert result.best_bid == 0.60
        assert result.best_ask == 0.70

    def test_returns_none_for_missing_question(self) -> None:
        market = {
            "questionID": "0x123abc",
            "clobTokenIds": json.dumps(["token_yes", "token_no"]),
            "outcomePrices": json.dumps(["0.65", "0.35"]),
        }
        result = _get_useful_market_info(market)
        assert result is None

    def test_returns_none_for_invalid_question_format(self) -> None:
        market = {
            "question": "Some other question format?",
            "questionID": "0x123abc",
            "clobTokenIds": json.dumps(["token_yes", "token_no"]),
            "outcomePrices": json.dumps(["0.65", "0.35"]),
        }
        result = _get_useful_market_info(market)
        assert result is None

    def test_handles_empty_tokens(self) -> None:
        market = {
            "question": "Will Amazon (AMZN) finish week of December 29 above $200?",
            "questionID": "0x123abc",
            "clobTokenIds": json.dumps([]),
            "outcomePrices": json.dumps([]),
        }
        result = _get_useful_market_info(market)
        assert result is not None
        assert result.yes_token_id is None
        assert result.no_token_id is None

    def test_handles_single_token(self) -> None:
        market = {
            "question": "Will Amazon (AMZN) finish week of December 29 above $200?",
            "questionID": "0x123abc",
            "clobTokenIds": json.dumps(["token_yes"]),
            "outcomePrices": json.dumps(["0.65"]),
        }
        result = _get_useful_market_info(market)
        assert result is not None
        assert result.yes_token_id == "token_yes"
        assert result.no_token_id is None


class TestFetchStockEvents:
    """Tests for fetch_stock_events function."""

    @patch("store.fetch_events.requests.Session")
    def test_fetches_matching_events(self, mock_session_cls: MagicMock) -> None:
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": "event1",
                "title": "Will Amazon (AMZN) finish week of December 29 above___?",
                "endDate": "2026-01-03T00:00:00Z",
                "markets": [
                    {
                        "question": "Will Amazon (AMZN) finish week of December 29 above $200?",
                        "questionID": "0x123",
                        "clobTokenIds": json.dumps(["yes_token", "no_token"]),
                        "outcomePrices": json.dumps(["0.6", "0.4"]),
                        "orderPriceMinTickSize": "0.01",
                        "negRisk": False,
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        result = fetch_stock_events()

        assert len(result) == 1
        assert result[0].question_id == "event1"
        assert result[0].symbol == "AMZN"
        assert result[0].markets is not None
        assert len(result[0].markets) == 1

    @patch("store.fetch_events.requests.Session")
    def test_filters_non_matching_events(self, mock_session_cls: MagicMock) -> None:
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": "event1",
                "title": "Will Bitcoin reach $100k?",
                "endDate": "2026-01-03T00:00:00Z",
                "markets": [],
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        result = fetch_stock_events()

        assert len(result) == 0

    @patch("store.fetch_events.requests.Session")
    def test_handles_pagination(self, mock_session_cls: MagicMock) -> None:
        # Pagination only continues when batch size equals BATCH_SIZE (500)
        # Since we return 1 event (< 500), it stops after first call
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": "event1",
                "title": "Will Amazon (AMZN) finish week of December 29 above___?",
                "endDate": "2026-01-03T00:00:00Z",
                "markets": [
                    {
                        "question": "Will Amazon (AMZN) finish week of December 29 above $200?",
                        "questionID": "0x123",
                        "clobTokenIds": json.dumps(["yes", "no"]),
                        "outcomePrices": json.dumps(["0.6", "0.4"]),
                    }
                ],
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        result = fetch_stock_events()

        assert len(result) == 1
        # Verify offset was passed in params
        call_args = mock_session.get.call_args
        assert call_args is not None
        assert "params" in call_args.kwargs or len(call_args.args) > 1

    @patch("store.fetch_events.requests.Session")
    def test_handles_empty_response(self, mock_session_cls: MagicMock) -> None:
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        result = fetch_stock_events()

        assert result == []

    @patch("store.fetch_events.requests.Session")
    def test_skips_events_with_unknown_symbols(self, mock_session_cls: MagicMock) -> None:
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": "event1",
                "title": "Will Unknown (UNKN) finish week of December 29 above___?",
                "endDate": "2026-01-03T00:00:00Z",
                "markets": [],
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        result = fetch_stock_events()

        assert len(result) == 0

    @patch("store.fetch_events.requests.Session")
    def test_raises_on_http_error(self, mock_session_cls: MagicMock) -> None:
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_session.get.return_value = mock_response

        with pytest.raises(Exception, match="HTTP Error"):
            fetch_stock_events()
