# smc_trading_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from pymongo import MongoClient

# -- Setup Binance Client (API_KEY, SECRET required)
clients = [Client("YOUR_API_KEY", "YOUR_SECRET")]

# -- Setup MongoDB
mongo = MongoClient("mongodb://localhost:27017/")
db = mongo["trading_db"]
positions_collection = db["positions"]
trades_collection = db["trades"]

# ---------------------------------------------
# Create Position with SL/TP/Trailing Stop/Order Type
# ---------------------------------------------
def create_position(symbol, side, quantity, price, sl=None, tp=None, trailing_percent=None, order_type="MARKET"):
    position = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "timestamp": datetime.utcnow(),
        "sl": sl,
        "tp": tp,
        "trailing_percent": trailing_percent,
        "order_type": order_type,
        "limit_order_id": None
    }

    client = clients[0]
    if order_type == "MARKET":
        order = client.create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
    else:  # LIMIT order
        order = client.create_order(
            symbol=symbol,
            side=side,
            type="LIMIT",
            quantity=quantity,
            timeInForce="GTC",
            price=str(price)
        )
        position["limit_order_id"] = order['orderId']
        position["limit_created_at"] = datetime.utcnow()

    positions_collection.insert_one(position)

# ---------------------------------------------
# AI Re-Entry Suggestion
# ---------------------------------------------
def ai_reentry_suggestion(symbol):
    df = get_binance_data(symbol, "15m")
    df['returns'] = df['close'].pct_change().fillna(0)
    df['volatility'] = df['returns'].rolling(window=10).std().fillna(0)
    df['direction'] = (df['returns'].shift(-1) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(df[['returns', 'volatility']].iloc[:-1], df['direction'].iloc[:-1])
    prediction = model.predict(df[['returns', 'volatility']].tail(1))
    return "BUY" if prediction[0] == 1 else "SELL"

# ---------------------------------------------
# Get Historical Binance Data
# ---------------------------------------------
def get_binance_data(symbol, interval="15m", lookback="100"):
    klines = clients[0].get_klines(symbol=symbol, interval=interval, limit=int(lookback))
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df

# ---------------------------------------------
# Chart Entry/SL/TP/Limit Visualizations
# ---------------------------------------------
def add_sl_tp_lines(fig, symbol):
    positions = list(positions_collection.find({"symbol": symbol}))
    for pos in positions:
        time = pd.to_datetime(pos["timestamp"])
        fig.add_trace(go.Scatter(
            x=[time],
            y=[pos['price']],
            mode='markers+text',
            marker=dict(color='orange', size=10),
            text=f"Entry ({pos['side']})",
            name=f"{pos['side']} Entry"
        ))
        if pos.get("sl"):
            fig.add_hline(y=pos["sl"], line=dict(color="red", dash="dash"), annotation_text="SL")
        if pos.get("tp"):
            fig.add_hline(y=pos["tp"], line=dict(color="green", dash="dash"), annotation_text="TP")
        if pos.get("order_type") == "LIMIT":
            fig.add_trace(go.Scatter(
                x=[time],
                y=[pos['price']],
                mode='markers',
                marker=dict(color='blue', symbol='x', size=8),
                name="Limit Order"
            ))
    return fig

# ---------------------------------------------
# Monitor SL/TP
# ---------------------------------------------
def monitor_positions():
    for pos in positions_collection.find():
        symbol = pos['symbol']
        side = pos['side']
        price = float(clients[0].get_symbol_ticker(symbol=symbol)['price'])
        sl = pos.get('sl')
        tp = pos.get('tp')

        if side == "BUY" and sl and price <= sl:
            positions_collection.delete_one({"_id": pos["_id"]})
        elif side == "BUY" and tp and price >= tp:
            positions_collection.delete_one({"_id": pos["_id"]})
        elif side == "SELL" and tp and price <= tp:
            positions_collection.delete_one({"_id": pos["_id"]})
        elif side == "SELL" and sl and price >= sl:
            positions_collection.delete_one({"_id": pos["_id"]})

# ---------------------------------------------
# Monitor and Auto-cancel Limit Orders
# ---------------------------------------------
def monitor_limit_orders():
    for pos in positions_collection.find({"order_type": "LIMIT", "limit_order_id": {"$ne": None}}):
        created_at = pos.get("limit_created_at")
        if created_at:
            elapsed = datetime.utcnow() - created_at
            if elapsed.total_seconds() > 300:
                try:
                    clients[0].cancel_order(symbol=pos['symbol'], orderId=pos['limit_order_id'])
                    positions_collection.delete_one({"_id": pos["_id"]})
                    print(f"⛔ Cancelled LIMIT order: {pos['limit_order_id']}")
                except Exception as e:
                    print("Limit Cancel Error:", e)

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="SMC AI Trading App", layout="wide")
st.markdown("<style>body { zoom: 80%; } @media screen and (max-width: 768px) { body { zoom: 60%; } }</style>", unsafe_allow_html=True)

st.title("📈 SMC + AI Price Action Trading Bot")
symbol = st.text_input("Symbol", "BTCUSDT")
col1, col2 = st.columns(2)
with col1:
    side = st.selectbox("Side", ["BUY", "SELL"])
    quantity = st.number_input("Quantity", min_value=0.001)
    order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
with col2:
    price = float(clients[0].get_symbol_ticker(symbol=symbol)['price'])
    sl = st.number_input("Stop Loss", min_value=0.0)
    tp = st.number_input("Take Profit", min_value=0.0)
    trailing = st.number_input("Trailing Stop %", min_value=0.0)

if st.button("📥 Create Position"):
    create_position(symbol, side, quantity, price, sl=sl, tp=tp, trailing_percent=trailing, order_type=order_type)
    st.success("✅ Position created!")

st.subheader("🤖 AI Re-Entry Suggestion")
ai_signal = ai_reentry_suggestion(symbol)
st.write(f"AI Suggests: **{ai_signal}**")

# ---------------------------------------------
# Show Price Chart
# ---------------------------------------------
data = get_binance_data(symbol, interval="15m")
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close']
)])
fig = add_sl_tp_lines(fig, symbol)
st.plotly_chart(fig, use_container_width=True)

# Run Monitors
monitor_positions()
monitor_limit_orders()
