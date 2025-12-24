"""Tests for orchestrator module."""

import asyncio

import pytest
from domain.models import OptionQuoteEvent
from store import EventStore, OptionStore
from trader.config import Config
from trader.orchestrator import Orchestrator


class TestOrchestrator:
    """Tests for Orchestrator class."""

    def test_initialization(self):
        """Test Orchestrator initializes with required config."""
        config = Config(polymarket_wallet="test", polymarket_api_key="test")
        orchestrator = Orchestrator(config=config)

        assert orchestrator.config == config
        assert orchestrator._running is False
        assert orchestrator._client is None
        assert orchestrator._listener is None

    def test_custom_stores(self):
        """Test Orchestrator accepts custom stores."""
        config = Config(polymarket_wallet="test", polymarket_api_key="test")
        option_store = OptionStore()
        event_store = EventStore()
        quote_queue: asyncio.Queue[OptionQuoteEvent] = asyncio.Queue(maxsize=100)

        orchestrator = Orchestrator(
            config=config,
            option_store=option_store,
            event_store=event_store,
            quote_queue=quote_queue,
        )

        assert orchestrator.option_store is option_store
        assert orchestrator.event_store is event_store
        assert orchestrator.quote_queue is quote_queue


class TestOrchestratorLifecycle:
    """Tests for Orchestrator lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self):
        """Test start() sets _running to True."""
        from unittest.mock import Mock, patch

        config = Config(polymarket_wallet="test", polymarket_api_key="test")
        orchestrator = Orchestrator(config=config)

        # Mock to avoid real network calls
        orchestrator.event_store.refresh = lambda: 0

        with patch("trader.orchestrator.PolymarketClient") as mock_client_cls:
            mock_client_cls.return_value = Mock()
            await orchestrator.start()
            assert orchestrator._running is True
            assert orchestrator._client is not None
            assert len(orchestrator._tasks) == 4

            # Cleanup
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_tasks(self):
        """Test stop() cancels all tasks and clears state."""
        from unittest.mock import Mock, patch

        config = Config(polymarket_wallet="test", polymarket_api_key="test")
        orchestrator = Orchestrator(config=config)
        orchestrator.event_store.refresh = lambda: 0

        with patch("trader.orchestrator.PolymarketClient") as mock_client_cls:
            mock_client_cls.return_value = Mock()
            await orchestrator.start()
            await orchestrator.stop()

            assert orchestrator._running is False
            assert len(orchestrator._tasks) == 0
