# Data Sources

This repository contains code only. Reproducing any result requires re-pulling the
underlying data from the sources below. None of the raw data is redistributed here
(NSE redistribution rules, file sizes, and subscription terms).

## NSE F&O Bhavcopy (free)

- URL: https://www.nseindia.com/all-reports
- Daily ZIP archives of F&O settlement data since 2003.
- Note: NSE changed the bhavcopy file-name and column format in mid-2024. The downloader
  handles both the legacy and current layouts.
- **Settlement-price gotcha:** on an option's expiry day the bhavcopy `settle_price`
  column carries the underlying index level, not the option premium. The panel builder
  corrects for this; a naive read overstates expiry-day premia by ~300×.

## DefinEdge API (subscription)

- Used for live and historical equity / F&O OHLC.
- Authentication: API key + secret in a local `.env`, plus an interactive TOTP on first
  login per session. Credentials and session tokens are **never** committed (see
  `.gitignore`).
- Expired F&O contracts are not retrievable through the high-level client; for historical
  expired options, source OHLC from the NSE bhavcopy archive instead.

## Yahoo Finance (free)

- Nifty 50 spot via ticker `^NSEI` (`yfinance`).
- Used for the daily index return series feeding MCMC estimation.

## RBI (free)

- 91-day Treasury-bill yields: https://rbi.org.in
- Used as the risk-free rate in option valuation and excess-return computation.

## IRDAI (free, P6 only)

- Insurer prospectuses for the ULIP study: https://irdai.gov.in

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017/` |
| `DEFINEDGE_API_KEY` | DefinEdge API key | (none — required for equity pulls) |
| `DEFINEDGE_API_SECRET` | DefinEdge API secret | (none — required for equity pulls) |

Set these in a local `.env` file (git-ignored). Never commit credentials.
