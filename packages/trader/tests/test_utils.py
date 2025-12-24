"""Tests for utility functions."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

from store import EventStore, OptionStore
from trader.utils import filter_expiries, get_unique_expiries, has_recent_option_data


class TestHasRecentOptionData:
    """Tests for has_recent_option_data function."""

    def test_returns_false_when_no_data(self):
        """Test returns False when store has no data."""
        option_store = OptionStore()
        assert has_recent_option_data(option_store) is False

    def test_returns_true_when_recent_data(self):
        """Test returns True when store was updated recently."""
        option_store = Mock(spec=OptionStore)
        option_store.last_write.return_value = datetime.now(UTC)
        assert has_recent_option_data(option_store) is True

    def test_returns_false_when_stale_data(self):
        """Test returns False when store data is stale."""
        option_store = Mock(spec=OptionStore)
        option_store.last_write.return_value = datetime.now(UTC) - timedelta(seconds=60)
        assert has_recent_option_data(option_store, max_age_seconds=30) is False


class TestGetUniqueExpiries:
    """Tests for get_unique_expiries function."""

    def test_returns_intersection_of_expiries(self):
        """Test returns sorted intersection of event and option expiries."""
        event_store = Mock(spec=EventStore)
        option_store = Mock(spec=OptionStore)

        expiry1 = datetime(2026, 1, 10, tzinfo=UTC)
        expiry2 = datetime(2026, 1, 17, tzinfo=UTC)
        expiry3 = datetime(2026, 1, 24, tzinfo=UTC)

        event_store.get_expiration_dates.return_value = {expiry1, expiry2}
        option_store.get_expiration_dates.return_value = {expiry2, expiry3}

        result = get_unique_expiries(event_store, option_store)
        assert result == [expiry2]

    def test_returns_empty_when_no_intersection(self):
        """Test returns empty list when no common expiries."""
        event_store = Mock(spec=EventStore)
        option_store = Mock(spec=OptionStore)

        expiry1 = datetime(2026, 1, 10, tzinfo=UTC)
        expiry2 = datetime(2026, 1, 17, tzinfo=UTC)

        event_store.get_expiration_dates.return_value = {expiry1}
        option_store.get_expiration_dates.return_value = {expiry2}

        result = get_unique_expiries(event_store, option_store)
        assert result == []


class TestFilterExpiries:
    """Tests for filter_expiries function."""

    def test_filters_by_max_days(self):
        """Test filters expiries to within max_days."""
        today = datetime.now(UTC).date()
        expiries = [
            datetime(today.year, today.month, today.day, tzinfo=UTC),
            datetime(today.year, today.month, today.day, tzinfo=UTC) + timedelta(days=7),
            datetime(today.year, today.month, today.day, tzinfo=UTC) + timedelta(days=14),
            datetime(today.year, today.month, today.day, tzinfo=UTC) + timedelta(days=21),
        ]

        result = filter_expiries(expiries, max_days=14)
        assert len(result) == 3  # Today + 7 days + 14 days

    def test_excludes_past_expiries(self):
        """Test excludes expiries before today."""
        today = datetime.now(UTC).date()
        expiries = [
            datetime(today.year, today.month, today.day, tzinfo=UTC) - timedelta(days=1),
            datetime(today.year, today.month, today.day, tzinfo=UTC),
        ]

        result = filter_expiries(expiries, max_days=14)
        assert len(result) == 1
