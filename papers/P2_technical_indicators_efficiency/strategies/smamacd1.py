# confluence_backtester.py
import backtrader as bt
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
import os

# --- MongoDB Configuration ---
# This should match your local setup.
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
MONGO_DATABASE_NAME = "price_data"

def load_tickers_from_file(filename):
    """Loads a list of tickers from a text file, one ticker per line."""
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        if not tickers:
            print(f"Warning: Ticker file '{filename}' is empty.")
            return []
        print(f"Successfully loaded {len(tickers)} tickers from '{filename}'.")
        return tickers
    except FileNotFoundError:
        print(f"ERROR: Ticker file not found at '{filename}'. Please create it.")
        return []

# Custom Data Feed to include the 'obv' column from your database
class PandasDataWithOBV(bt.feeds.PandasData):
    lines = ('obv',)
    params = (
        ('obv', -1), # -1 means auto-detect column by name
    )

class TrendMomentumVolumeStrategy(bt.Strategy):
    """
    A strategy that requires confluence from Trend, Momentum, Volume, and Volatility filters.
    """
    params = (
        ('sma_period', 200),
        ('ema_fast', 12),
        ('ema_slow', 26),
        ('atr_period', 14),
        ('atr_ma_period', 20),
        ('stop_loss_atr_multiplier', 2.0),
    )

    def __init__(self):
        # Data references
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        self.obv = self.datas[0].obv

        # Trend Filter
        self.sma200 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.p.sma_period)

        # Momentum Filter
        self.ema_fast = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.p.ema_fast)
        self.ema_slow = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.p.ema_slow)
        self.macd = bt.indicators.MACD(self.datas[0], period_me1=self.p.ema_fast, period_me2=self.p.ema_slow)
        self.macd_hist = self.macd.macd - self.macd.signal

        # Volatility Filter
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=self.p.atr_period)
        self.atr_ma = bt.indicators.SimpleMovingAverage(self.atr, period=self.p.atr_ma_period)

        # Order tracking
        self.order = None
        self.stop_loss_order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()}, {txt}') # Suppressed for batch runs

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy(): self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size}')
            elif order.issell(): self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, Size: {order.executed.size}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected for {order.data._name}')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        if self.order: return
        if not self.position:
            is_uptrend = self.dataclose[0] > self.sma200[0]
            is_momentum_up = self.ema_fast[0] > self.ema_slow[0] and self.macd_hist[0] > 0
            is_volume_confirming_up = self.obv[0] > self.obv[-1]
            is_volatility_surging = self.atr[0] > self.atr_ma[0]
            if is_uptrend and is_momentum_up and is_volume_confirming_up and is_volatility_surging:
                self.log(f'LONG ENTRY SIGNAL, {self.dataclose[0]:.2f}')
                size = int(self.broker.get_cash() * 0.20 / self.dataclose[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    stop_price = self.dataclose[0] - self.p.stop_loss_atr_multiplier * self.atr[0]
                    self.stop_loss_order = self.sell(exectype=bt.Order.Stop, price=stop_price, size=size)
        else:
            if self.position.size > 0 and self.ema_fast[0] < self.ema_slow[0]:
                self.log(f'LONG EXIT (MOMENTUM REVERSAL), {self.dataclose[0]:.2f}')
                self.order = self.close()
                self.broker.cancel(self.stop_loss_order)

def run_backtest_from_mongo(ticker, mongo_client, db_name, start_date, end_date, initial_cash=100000.0, commission=0.001):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(TrendMomentumVolumeStrategy)

    try:
        db = mongo_client[db_name]
        collection = db[ticker]
        
        cursor = collection.find({})
        data = pd.DataFrame(list(cursor))
        
        if data.empty:
            print(f"  -> No data found for {ticker} in the collection. Skipping.")
            return None, None

        date_col = 'convertedDate' if 'convertedDate' in data.columns else 'Date'
        data.rename(columns={date_col: 'datetime'}, inplace=True)
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

        if data.empty:
            print(f"  -> No data found for {ticker} in the specified date range. Skipping.")
            return None, None
            
        # UPDATED: Add check for sufficient data length
        min_data_length = 200 # Based on the longest indicator period (SMA_200)
        if len(data) < min_data_length:
            print(f"  -> Insufficient data for {ticker} ({len(data)} rows). Need at least {min_data_length}. Skipping.")
            return None, None

        data.set_index('datetime', inplace=True)
        data.columns = [col.lower() for col in data.columns]
        
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'obv']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data for {ticker}")

        feed = PandasDataWithOBV(dataname=data, name=ticker)
        cerebro.adddata(feed)

    except Exception as e:
        print(f"  -> Error loading data for {ticker} from MongoDB: {e}. Skipping.")
        return None, None

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=252)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')

    results = cerebro.run()
    strat = results[0]
    trade_analysis = strat.analyzers.tradeanalyzer.get_analysis()
    
    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_cash
    
    if trade_analysis.total.total > 0:
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        ann_ret = strat.analyzers.returns.get_analysis()['rnorm100']
    else:
        sharpe, max_dd, ann_ret, pnl, final_value = 0.0, 0.0, 0.0, 0.0, initial_cash

    summary = {
        'Ticker': ticker, 'Final Value': final_value, 'Net PnL': pnl,
        'Sharpe Ratio': sharpe, 'Max Drawdown (%)': max_dd,
        'Annualized Return (%)': ann_ret, 'Total Trades': trade_analysis.total.total
    }

    trades_list = []
    if 'trades' in trade_analysis:
        for t in trade_analysis.trades:
            trade_info = {
                'Ticker': ticker, 'Direction': 'Long', 'Entry Date': t.open_datetime().strftime('%Y-%m-%d'),
                'Entry Price': t.price, 'Exit Date': t.close_datetime().strftime('%Y-%m-%d'),
                'Exit Price': t.exitprice, 'Size': t.size, 'PnL': t.pnl, 'PnL (Net)': t.pnlcomm
            }
            trades_list.append(trade_info)
    
    trade_log_df = pd.DataFrame(trades_list)
    return summary, trade_log_df

if __name__ == '__main__':
    TICKER_FILE = 'allticker.txt'
    tickers_to_test = load_tickers_from_file(TICKER_FILE)
    
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=18 * 30)
    
    all_summaries = []
    all_trade_logs = {}

    if not tickers_to_test:
        exit()

    try:
        client = MongoClient(MONGO_CONNECTION_STRING)
        client.admin.command('ismaster')
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to MongoDB. Please check your connection string.")
        print(f"Details: {e}")
        exit()

    print(f"\nStarting batch backtest for {len(tickers_to_test)} tickers...")
    
    for i, ticker in enumerate(tickers_to_test):
        print(f"\n({i+1}/{len(tickers_to_test)}) Running backtest for: {ticker}")
        summary, trade_log = run_backtest_from_mongo(
            ticker=ticker,
            mongo_client=client,
            db_name=MONGO_DATABASE_NAME,
            start_date=START_DATE,
            end_date=END_DATE,
            initial_cash=100000.0
        )
        if summary:
            all_summaries.append(summary)
            if not trade_log.empty:
                all_trade_logs[ticker] = trade_log
            print(f"  -> Completed. Net PnL: {summary['Net PnL']:.2f}, Trades: {summary['Total Trades']}")

    client.close()

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        output_filename = "confluence_backtest_summary.xlsx"
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
            for ticker, trade_df in all_trade_logs.items():
                sanitized_ticker = ticker.replace('.NS', '').replace('-', '_')[:31]
                trade_df.to_excel(writer, sheet_name=f'{sanitized_ticker}_Trades', index=False)
        print("\n--- Batch backtest complete! ---")
        print(f"Summary results saved to '{output_filename}'")
    else:
        print("\n--- Batch backtest complete! ---")
        print("No results were generated.")
