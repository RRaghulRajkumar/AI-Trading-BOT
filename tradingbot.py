'''Simple trading bot that uses the finbert sentiment analysis model to trade SPY.'''

'''
lumibot - Algorithm trading framework
alpaca-trade-api-python - getting news and place traders to broker
datetime - calculating data difference
torch - pytorch framework for using AI/ML
transformers - load up finance deep learning model

'''

import os
from datetime import datetime, timezone
from alpaca_trade_api import REST
from dotenv import load_dotenv
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from timedelta import Timedelta

from finbert_utils import estimate_sentiment

load_dotenv()  # take environment variables from .env.
# Retrieve API key and secret from environment variables

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_CREDS = {"API_KEY": API_KEY, "API_SECRET": API_SECRET, "PAPER": True}


PROBABILITY_THRESHOLD = 0.999
TOKENIZERS_PARALLELISM = False

class MLTrader(Strategy):
    """Strategy that uses finbert sentiment analysis to trade SPY."""

    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.5):
        """Initialize the strategy."""
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        """Calculate position size."""
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        """Get today's date and the date three days prior."""
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

    def get_sentiment(self):
        """Get sentiment from finbert model."""
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        """Execute trading logic."""
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > PROBABILITY_THRESHOLD:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95,
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > PROBABILITY_THRESHOLD:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05,
                )
                self.submit_order(order)
                self.last_trade = "sell"


start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2023, 12, 31, tzinfo=timezone.utc)
broker = Alpaca(ALPACA_CREDS)
strategy = MLTrader(
    name="mlstrat",
    broker=broker,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5},
)
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5},
)
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
