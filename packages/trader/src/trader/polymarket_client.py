"""Polymarket CLOB API client for executing trades."""

import asyncio
import logging

import requests
from domain.models import MarketMetadata, TradeDecision
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import CreateOrderOptions, OrderArgs, OrderType, PostOrdersArgs, SignedOrder

logger = logging.getLogger(__name__)


class PolymarketClient:
    """Client for interacting with Polymarket CLOB API.

    Handles authentication, order construction, and API communication.
    Uses the official py-clob-client SDK.
    """

    def __init__(self, api_key: str, wallet_address: str):
        """Initialize Polymarket client.

        Args:
            api_key: Private key for signing transactions (hex string with or without 0x prefix)
            wallet_address: Main wallet address (for Browser Wallet/Proxy setup)
        """
        private_key = api_key.replace("0x", "") if api_key.startswith("0x") else api_key

        self._client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            signature_type=1,
            funder=wallet_address,
        )
        self._http = requests.Session()

        try:
            creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)
            logger.info("Polymarket client authenticated successfully")
        except Exception as e:
            logger.error(f"Failed to derive API credentials: {e}")
            raise RuntimeError("Polymarket client authentication failed") from e

    def check_connection(self) -> bool:
        """Verify connection and authentication by fetching API keys.

        Returns:
            bool: True if authenticated and connected, False otherwise.
        """
        try:
            self._client.get_api_keys()
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    def get_fill_and_kill_order_object(self, signed_order: SignedOrder) -> PostOrdersArgs:
        """Create a FAK order object with order and order type from a signed order.

        Args:
            signed_order: Signed order object.

        Returns:
            PostOrdersArgs: Post FAK order object.
        """
        return PostOrdersArgs(order=signed_order, orderType=OrderType.FAK)  # type: ignore[arg-type]

    def get_signed_order(self, market: MarketMetadata, decision: TradeDecision) -> SignedOrder:
        """
        Creates and signs an order
        Level 1 Auth required
        """
        self._client.assert_level_1_auth()

        if decision.outcome == "YES":
            token_id = market.yes_token_id
        elif decision.outcome == "NO":
            token_id = market.no_token_id
        else:
            raise ValueError(f"Invalid outcome: {decision.outcome}")

        if token_id is None:
            raise ValueError(f"Missing token_id for outcome {decision.outcome}")
        if decision.price is None or decision.size is None or decision.side is None:
            raise ValueError("Decision missing required fields: price, size, or side")

        price = round(decision.price, 2)
        fee_rate_bps = market.fee_rate_bps if market.fee_rate_bps is not None else 0
        tick_size = market.tick_size if market.tick_size is not None else "0.01"
        neg_risk = market.neg_risk if market.neg_risk is not None else False

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=decision.size,
            side=decision.side,
            fee_rate_bps=fee_rate_bps,
        )

        order_options = CreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        return self._client.builder.create_order(
            order_args=order_args,
            options=order_options,
        )

    def build_order(self, market: MarketMetadata, decision: TradeDecision) -> PostOrdersArgs:
        """Build FAK order object from MarketMetadata and TradeDecision.

        Combines order signing and FAK object creation into a single call.

        Args:
            market: Market metadata with token IDs and pricing info.
            decision: Trade decision with side, outcome, price, and size.

        Returns:
            PostOrdersArgs: Ready-to-submit FAK order object.
        """
        signed_order = self.get_signed_order(market=market, decision=decision)
        return self.get_fill_and_kill_order_object(signed_order=signed_order)

    async def place_orders_async(self, orders: list[PostOrdersArgs]) -> list[dict]:
        return await asyncio.to_thread(self.place_orders, orders)

    def place_orders(self, orders: list[PostOrdersArgs]) -> list[dict]:
        response = self._client.post_orders(args=orders)
        return response if isinstance(response, list) else []
