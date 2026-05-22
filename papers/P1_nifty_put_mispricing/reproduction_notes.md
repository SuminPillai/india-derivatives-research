# P1 Reproduction Notes

## Environment

- Python 3.13, dependencies pinned in the top-level `requirements.txt`.
- Local MongoDB (`MONGO_URI`, default `mongodb://localhost:27017/`). The options panel and
  spot/T-bill series are staged in MongoDB before estimation.
- `np.random.seed(20260509)` is fixed in every script.

## Inputs and expected outputs

| Script | Input | Output |
|--------|-------|--------|
| `00_nifty_spot_and_tbill.py` | Yahoo `^NSEI`, RBI 91-day T-bill | Daily spot + risk-free series in MongoDB |
| `01_bhavcopy_downloader.py` | NSE bhavcopy ZIPs | Parsed F&O records (`optionsdatabase`) |
| `02_nifty_options_panel.py` | Parsed records | Held-to-maturity put panel, k = 0.94–1.00 |
| `03_mcmc_estimation.py` | Daily Nifty returns | BS / SV / SVJ structural parameters |
| `04_option_simulation.py` | Parameters + panel | 25,000-path p-values per bucket |
| `05_regime_split.py` | Panel + parameters | Pre-2018 / NBFC / COVID / inflation sub-period results |

## Known caveats

- **SVJ MCMC convergence.** The SVJ chains did not fully converge; SVJ parameters are
  reported via method of moments and treated as a robustness check on the SV results, not
  as the primary specification. This is stated in the paper and the abstract.
- **Bhavcopy settlement-price gotcha.** On expiry day, the NSE bhavcopy `settle_price`
  column carries the index level, not the option premium. `02_nifty_options_panel.py`
  corrects for this; bypassing the correction overstates expiry-day premia by ~300×.
- **Bhavcopy format change.** NSE changed the file naming/columns in mid-2024; the
  downloader handles both layouts.

## Sample

119 monthly expiry cycles, January 2015 – April 2025. Regime split: pre-2018 (35 months),
NBFC stress (23), COVID (24), inflation/rate-hike (37).
