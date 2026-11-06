"""SmartGrid Campus - AI energy manager (Track 5). Run: streamlit run app.py"""
import numpy as np, pandas as pd, streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="SmartGrid Campus", page_icon="⚡", layout="wide")
# name: (base kW, working-hour ranges, runs on weekends, max share that can be shed at peak)
ZONES = {"Block A – Classrooms": (40, [(9, 16)], False, .35), "Block B – Labs": (55, [(10, 17)], False, .12),
         "Library": (25, [(8, 20)], True, .20), "Hostel": (60, [(5, 8), (18, 23)], True, .30),
         "Admin Office": (20, [(9, 17)], False, .30)}

@st.cache_data
def demo():
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=24 * 14, freq="h")
    rng = np.random.default_rng(7); waste = {d: rng.random() < .3 for d in idx.date}; out = {}
    for z, (base, rs, wk, _) in ZONES.items():
        v = []
        for ts in idx:
            on = any(a <= ts.hour < b for a, b in rs) and (wk or ts.dayofweek < 5)
            lvl = .9 if on else (.45 if waste[ts.date()] else .15)
            v.append(base * lvl * (1 + rng.normal(0, .04)))
        out[z] = v
    df = pd.DataFrame(out, index=idx); df.iloc[-6:, 1] *= 2  # simulated fault in the labs
    return df.round(1)

def load(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce")
    return df.sort_index().resample("h").mean().interpolate().dropna(how="all")

def feats(ix):
    return pd.DataFrame({"h": ix.hour, "dow": ix.dayofweek, "s": np.sin(2 * np.pi * ix.hour / 24),
                         "c": np.cos(2 * np.pi * ix.hour / 24)}, index=ix)

def forecast(total):
    X = feats(total.index); X["lag"] = total.shift(24).values; ok = X["lag"].notna()
    m = GradientBoostingRegressor(random_state=0).fit(X[ok].iloc[:-24], total[ok].iloc[:-24])
    hold = pd.Series(m.predict(X[ok].iloc[-24:]), index=X[ok].index[-24:])
    mae = (hold - total[hold.index]).abs().mean()
    m.fit(X[ok], total[ok])
    fx = total.index[-1] + pd.to_timedelta(range(1, 25), unit="h"); F = feats(fx)
    F["lag"] = total.reindex(fx - pd.Timedelta(hours=24)).values
    return pd.Series(m.predict(F), index=fx), mae

st.title("⚡ SmartGrid Campus")
st.caption("AI energy management: monitor, forecast, cut waste and shave peaks.")
with st.sidebar:
    st.header("Settings")
    up = st.file_uploader("Upload meter data (CSV)", type="csv", help="First column: timestamp. One column per zone, in kW.")
    limit = st.slider("Peak limit (kW)", 80, 300, 125)
    tariff = st.number_input("Tariff (₹ per kWh)", 1.0, 30.0, 8.0, .5)
    ef = st.number_input("CO₂ factor (kg per kWh)", .1, 1.5, .82, .01)
    st.download_button("Download CSV template", demo().head(48).to_csv(), "template.csv")
try:
    df = load(up) if up else demo()
except Exception as e:
    st.error(f"Could not read the file: {e}"); st.stop()
if len(df) < 72:
    st.error("Need at least 3 days of hourly data."); st.stop()
st.info("Using uploaded data." if up else "Showing simulated demo data. Upload your own CSV in the sidebar.")
total = df.sum(axis=1); kwh = total.sum()
t1, t2, t3, t4 = st.tabs(["📊 Overview", "🔮 Forecast & peaks", "💡 Savings", "🚨 Anomalies"])

with t1:
    c = st.columns(4)
    c[0].metric("Energy used", f"{kwh:,.0f} kWh"); c[1].metric("Peak load", f"{total.max():.0f} kW", f"at {total.idxmax():%d %b %H:%M}", delta_color="off")
    c[2].metric("Estimated cost", f"₹{kwh * tariff:,.0f}"); c[3].metric("CO₂ emitted", f"{kwh * ef:,.0f} kg")
    st.subheader("Campus load (kW)"); st.line_chart(total.rename("Total load"))
    st.subheader("Energy by zone (kWh)"); st.bar_chart(df.sum())

with t2:
    fc, mae = forecast(total)
    st.caption(f"Model: gradient boosting on hour, weekday and last-day load. Error on the last 24 h: {mae:.1f} kW.")
    ch = pd.concat([total.iloc[-72:].rename("Actual"), fc.rename("Forecast")], axis=1); ch["Limit"] = limit
    st.line_chart(ch)
    over = fc[fc > limit]
    if over.empty:
        st.success(f"No peak above {limit} kW forecast for the next 24 hours.")
    else:
        st.warning(f"Peak above {limit} kW forecast at {len(over)} hour(s). Suggested load shedding:")
        prof = df.groupby(df.index.hour).mean(); rows = []
        for ts, v in over.items():
            pot = {z: prof.loc[ts.hour, z] * ZONES.get(z, (0, 0, 0, .2))[3] for z in df.columns}
            need = v - limit; s = sum(pot.values()); r = {"Hour": f"{ts:%a %H:%M}", "Forecast kW": round(v), "Reduce by kW": round(need)}
            r.update({z: round(min(need, s) * p / s, 1) for z, p in pot.items()}); r["Enough?"] = "Yes" if s >= need else "Partly"
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.download_button("Download forecast CSV", fc.rename("forecast_kW").to_csv(), "forecast.csv")

with t3:
    st.subheader("Idle-load waste by zone")
    prof = df.groupby(df.index.hour).median(); rows = []
    for z in df.columns:
        busy = prof[z] > .5 * prof[z].max(); idle = ~df.index.hour.isin(busy[busy].index)
        stby = df[z].quantile(.10); w = (df[z][idle] - 1.5 * stby).clip(lower=0).sum()
        rows.append({"Zone": z, "Standby kW": round(stby, 1), "Avoidable kWh": round(w), "Cost ₹": round(w * tariff), "CO₂ kg": round(w * ef)})
    r = pd.DataFrame(rows); st.dataframe(r, width="stretch", hide_index=True)
    st.success(f"Switching off idle loads could avoid up to {r['Avoidable kWh'].sum():,.0f} kWh ({100 * r['Avoidable kWh'].sum() / kwh:.1f}% of use), about ₹{r['Cost ₹'].sum():,.0f}.")
    st.caption("Avoidable = usage outside working hours above 1.5× the zone's normal standby level.")

with t4:
    st.subheader("Abnormal consumption (last 7 days)")
    rows = []
    for z in df.columns:
        g = df[z].groupby([df.index.dayofweek >= 5, df.index.hour]); med = g.transform("median")
        sd = (df[z] - med).abs().groupby([df.index.dayofweek >= 5, df.index.hour]).transform("median") * 1.4826
        sd = sd.clip(lower=.05 * df[z].mean()); zs = (df[z] - med) / sd
        for ts in df.index[(zs > 4) & ((df[z] - med) > df[z].mean()) & (df.index > df.index[-1] - pd.Timedelta(days=7))]:
            rows.append({"Time": ts, "Zone": z, "Actual kW": round(df.at[ts, z], 1), "Expected kW": round(med[ts], 1), "Excess kW": round(df.at[ts, z] - med[ts], 1)})
    if rows: st.error(f"{len(rows)} abnormal reading(s) found. Check these circuits."); st.dataframe(pd.DataFrame(rows).sort_values("Time", ascending=False), width="stretch", hide_index=True)
    else: st.success("No abnormal consumption detected.")
