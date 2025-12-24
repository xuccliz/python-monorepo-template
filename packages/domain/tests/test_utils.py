"""Tests for domain utility functions."""

from datetime import UTC, datetime

import pytest
from domain.utils import make_expiry_datetime, time_to_expiry_years


class TestMakeExpiryDatetime:
    """Tests for make_expiry_datetime function."""

    def test_creates_datetime_with_market_close_time(self) -> None:
        """Test that datetime is created with 21:00 UTC (4PM ET market close)."""
        result = make_expiry_datetime("2026-01-17")

        assert result.year == 2026
        assert result.month == 1
        assert result.day == 17
        assert result.hour == 21
        assert result.minute == 0
        assert result.second == 0
        assert result.tzinfo == UTC

    def test_different_dates(self) -> None:
        """Test with various date strings."""
        result = make_expiry_datetime("2025-12-31")
        assert result == datetime(2025, 12, 31, 21, 0, 0, tzinfo=UTC)

        result = make_expiry_datetime("2024-06-15")
        assert result == datetime(2024, 6, 15, 21, 0, 0, tzinfo=UTC)


class TestTimeToExpiryYears:
    """Tests for time_to_expiry_years function."""

    def test_calculates_time_to_expiry(self) -> None:
        """Test time to expiry calculation."""
        now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        expiry = datetime(2027, 1, 1, 0, 0, 0, tzinfo=UTC)

        result = time_to_expiry_years(expiry, now)

        # Should be approximately 1 year
        assert result == pytest.approx(1.0, rel=0.01)

    def test_uses_current_time_when_now_is_none(self) -> None:
        """Test that current time is used when now is None."""
        # Use a far future expiry to ensure positive result
        expiry = datetime(2030, 1, 1, 0, 0, 0, tzinfo=UTC)

        result = time_to_expiry_years(expiry)

        # Should be positive (future expiry)
        assert result > 0

    def test_negative_time_for_past_expiry(self) -> None:
        """Test that past expiry returns negative time."""
        now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        expiry = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)

        result = time_to_expiry_years(expiry, now)

        assert result < 0
        assert result == pytest.approx(-1.0, rel=0.01)

    def test_zero_time_when_same(self) -> None:
        """Test that same time returns zero."""
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

        result = time_to_expiry_years(now, now)

        assert result == 0.0


class TestIsMarketOpen:
    """Tests for is_market_open function."""

    def test_market_closed_on_weekend(self) -> None:
        """Test that market is closed on Saturday."""
        from domain.utils import is_market_open

        # Saturday at noon UTC
        saturday = datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)
        assert is_market_open(saturday) is False

    def test_market_closed_on_sunday(self) -> None:
        """Test that market is closed on Sunday."""
        from domain.utils import is_market_open

        # Sunday at noon UTC
        sunday = datetime(2026, 1, 4, 12, 0, 0, tzinfo=UTC)
        assert is_market_open(sunday) is False

    def test_market_open_during_trading_hours(self) -> None:
        """Test that market is open during regular trading hours."""
        from domain.utils import is_market_open

        # Monday at 15:00 UTC (10 AM ET) - market should be open
        monday_trading = datetime(2026, 1, 5, 15, 0, 0, tzinfo=UTC)
        assert is_market_open(monday_trading) is True

    def test_market_closed_before_open(self) -> None:
        """Test that market is closed before opening bell."""
        from domain.utils import is_market_open

        # Monday at 13:00 UTC (8 AM ET) - before market open
        monday_early = datetime(2026, 1, 5, 13, 0, 0, tzinfo=UTC)
        assert is_market_open(monday_early) is False

    def test_market_closed_after_close(self) -> None:
        """Test that market is closed after closing bell."""
        from domain.utils import is_market_open

        # Monday at 22:00 UTC (5 PM ET) - after market close
        monday_late = datetime(2026, 1, 5, 22, 0, 0, tzinfo=UTC)
        assert is_market_open(monday_late) is False

    def test_uses_current_time_when_none(self) -> None:
        """Test that current time is used when dt is None."""
        from domain.utils import is_market_open

        # Should not raise, returns bool
        result = is_market_open()
        assert isinstance(result, bool)


class TestGetNextMarketOpen:
    """Tests for get_next_market_open function."""

    def test_returns_next_open_from_weekend(self) -> None:
        """Test getting next market open from a weekend."""
        from domain.utils import get_next_market_open

        # Saturday
        saturday = datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)
        next_open = get_next_market_open(saturday)

        assert next_open is not None
        # Should be Monday
        assert next_open.weekday() == 0  # Monday

    def test_returns_next_open_from_after_hours(self) -> None:
        """Test getting next market open from after hours."""
        from domain.utils import get_next_market_open

        # Monday at 22:00 UTC (after close)
        monday_late = datetime(2026, 1, 5, 22, 0, 0, tzinfo=UTC)
        next_open = get_next_market_open(monday_late)

        assert next_open is not None
        # Should be Tuesday
        assert next_open.weekday() == 1  # Tuesday

    def test_uses_current_time_when_none(self) -> None:
        """Test that current time is used when dt is None."""
        from domain.utils import get_next_market_open

        result = get_next_market_open()
        # Should return a datetime or None
        assert result is None or isinstance(result, datetime)
