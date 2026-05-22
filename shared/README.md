# Shared Resources

Cross-paper resources used by more than one study in this series.

## `event_timeline_2015_2025.csv`

A timeline of SEBI/RBI regulatory actions and macro shocks (April 2015 – March 2026) that
plausibly affected Nifty 50 implied volatility, realized volatility, the India VIX, and
option-market microstructure. It supports the regime/sub-period splitting in P1 and P3.

**Status: placeholder header only.** The compiled timeline currently lives as a narrative
research document and has not yet been reduced to a clean machine-readable CSV. The header
row defines the intended schema:

| Column | Meaning |
|--------|---------|
| `date` | Event date (YYYY-MM-DD) |
| `event_name` | Short event label |
| `category` | Regulatory (SEBI/RBI), macro shock, or structural break |
| `description` | One-sentence description |
| `affected_segment` | Market segment (equity, debt, derivatives, all) |
| `source_url` | Primary source |
| `vol_impact_direction` | Expected direction of volatility impact (up/down/ambiguous) |

To populate it, transcribe the dated rows from the source research report into this schema.
