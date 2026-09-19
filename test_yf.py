import yfinance as yf
print("XAU:", len(yf.Ticker("XAU").history(period="1y")))
print("XAUUSD=X:", len(yf.Ticker("XAUUSD=X").history(period="1y")))
print("GC=F:", len(yf.Ticker("GC=F").history(period="1y")))
print("XAGUSD=X:", len(yf.Ticker("XAGUSD=X").history(period="1y")))
