# ISO-NE demand data sources

## Options identified

1. **Web Services API — Real-Time Hourly Demand**
   `GET /api/v1.1/realtimehourlydemand/day/{yyyymmdd}/location/{locationId}`
   The physically-meaningful "actual load" series. This is the primary
   target series (`timestamp | zone | demand_MW`).

2. **Web Services API — Day-Ahead Cleared Demand**
   `GET /api/v1.1/dayaheadhourlydemand/day/{yyyymmdd}/location/{locationId}`
   Not actual load (it's the cleared day-ahead market quantity), but cheap to
   fetch alongside real-time in the same job, so we keep it rather than
   discard it, per the source instructions. Useful later as a covariate or
   sanity check, not as the primary demand label.

3. **ISO Express web reports** (`iso-ne.com/isoexpress/web/reports/load-and-demand/`)
   Browser-based CSV report downloads (e.g. "SMD Hourly Data", "Daily Summary
   of Hourly Data", "Monthly Data by Load Zone"). These may require
   CAPTCHA/session-based access rather than a clean scriptable pull, so the
   Web Services API is preferred for reproducibility. Worth a manual spot
   check as a cross-validation source, not the primary download path.

Both API endpoints require a free ISO Express account (HTTP Basic Auth over
SSL) -- register at https://www.iso-ne.com/isoexpress, then set:

```bash
export ISONE_WS_USERNAME="..."
export ISONE_WS_PASSWORD="..."
```

## Zones

Location IDs for the 8 target zones (Maine, New Hampshire, Vermont,
Connecticut, Rhode Island, NEMA, SEMA, WCMA) are **not hardcoded** --
`src/isone/download_isone_demand.py` discovers them at runtime from
`GET /api/v1.1/locations.json` by matching zone names, since the numeric IDs
aren't reliably documented and a wrong guess would silently pull the wrong
zone's data.

## Timezone / DST

ISO-NE reports are understood to be published in Eastern time (with DST).
**To confirm and document precisely once real data is in hand:**
- [ ] What timezone/convention the API's `BeginDate`/hour fields actually use.
- [ ] How the "2 AM" and "duplicate hour" DST transition days are represented
      (e.g. an extra/missing hour, or a flagged duplicate).
- [ ] Store both a UTC timestamp and the original local timestamp explicitly
      -- do not silently convert one into the other (per source instructions).

## Recorded findings (fill in after first real pull)

(empty)
