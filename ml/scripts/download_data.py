"""
Download historical market data from Binance.

Usage:
    python download_data.py --symbol WBNBUSDT --interval 1m --days 60
"""

import argparse
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_klines(
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
    output_path: str
):
    """
    Download OHLCV data from Binance.

    Args:
        symbol: Trading pair (e.g., 'WBNBUSDT')
        interval: Candle interval (e.g., '1m', '5m', '1h')
        start_date: Start datetime
        end_date: End datetime
        output_path: Where to save the data
    """
    print(f"Downloading {symbol} {interval} data from {start_date} to {end_date}...")

    # Initialize Binance client (no API key needed for public data)
    client = Client()

    # Download klines
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_date.strftime('%d %b %Y'),
        end_str=end_date.strftime('%d %b %Y')
    )

    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Keep only needed columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Downloaded {len(df):,} candles")
    print(f"Saved to {output_path}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def download_order_book_snapshot(
    symbol: str,
    output_path: str,
    limit: int = 100
):
    """
    Download current order book snapshot.

    Note: Historical order book data requires paid subscription.
    This downloads current snapshot for reference.

    Args:
        symbol: Trading pair
        output_path: Where to save
        limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)
    """
    print(f"Downloading order book snapshot for {symbol}...")

    client = Client()
    depth = client.get_order_book(symbol=symbol, limit=limit)

    # Convert to DataFrame
    bids_df = pd.DataFrame(depth['bids'], columns=['price', 'quantity'])
    asks_df = pd.DataFrame(depth['asks'], columns=['price', 'quantity'])

    bids_df['side'] = 'bid'
    asks_df['side'] = 'ask'

    df = pd.concat([bids_df, asks_df], ignore_index=True)
    df['price'] = df['price'].astype(float)
    df['quantity'] = df['quantity'].astype(float)
    df['timestamp'] = datetime.now()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved order book snapshot to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Download Binance market data')
    parser.add_argument('--symbol', type=str, default='WBNBUSDT', help='Trading pair')
    parser.add_argument('--interval', type=str, default='1m', help='Candle interval')
    parser.add_argument('--days', type=int, default=60, help='Number of days to download')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--download-orderbook', action='store_true', help='Download order book snapshot')

    args = parser.parse_args()

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Download OHLCV data
    klines_output = os.path.join(
        args.output_dir,
        f"{args.symbol}_{args.interval}_{args.days}d.csv"
    )
    download_klines(args.symbol, args.interval, start_date, end_date, klines_output)

    # Download order book snapshot (for reference)
    if args.download_orderbook:
        orderbook_output = os.path.join(
            args.output_dir,
            f"{args.symbol}_orderbook_snapshot.csv"
        )
        download_order_book_snapshot(args.symbol, orderbook_output)

    print("\nDownload complete!")
    print(f"Next step: python scripts/train_xgboost.py --data {klines_output}")


if __name__ == '__main__':
    main()
