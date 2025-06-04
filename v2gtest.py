# 🔋 V2G-Simulator 2024
# -----------------------------------
import streamlit as st, pandas as pd, numpy as np, pulp, matplotlib.pyplot as plt
from pathlib import Path
CODE_VERSION = "1.0.3"          # cache-sleutel
# ─────────────────────────────── 0a  Bestands-paden
PRICE_PATH = "Stroomprijzen_2024_Met_Weekinfo.xlsx"
PV_PATH    = "pv_uurdata_2024.xlsx"
USE_PATH   = "Elektriciteitsgebruik per uur.xlsx"
# ───────────────────────────────   Helpers
@st.cache_data(show_spinner=False, ttl=0)  
def sheet_names(src): return pd.ExcelFile(src).sheet_names
@st.cache_data
def _drop_unnamed(df): return df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
def _standardize_index(df):
    if hasattr(df.index,'tz') and df.index.tz is not None: df.index = df.index.tz_localize(None)
    return df
@st.cache_data
def load_prices(src):
    df = pd.read_excel(src, parse_dates=["datum"])
    df = _drop_unnamed(df).set_index("datum")
    for c in ("Inkoop","Verkoop"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    return _standardize_index(df)
@st.cache_data
def load_profile(src,sheet,col):
    df = pd.read_excel(src, sheet_name=sheet, parse_dates=["Datum"])
    df = _drop_unnamed(df).set_index("Datum")
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    if df[col].max()>100: df[col]/=1000
    return _standardize_index(df).rename(columns={col:col.lower()})
# ─────────────────────────────── 0  UI-meta
st.set_page_config(layout="wide", page_title="🔋 V2G Simulator 2024")
st.title("🔋 V2G Simulator 2024")
st.markdown("""

Deze simulator berekent hoeveel u met uw elektrische auto, zonnepanelen en een bidirectionele laadpaal kunt besparen op uw energierekening!

Om een zo realistisch mogelijke berekening te maken, vult u het volgende in:
- De capaciteit en laadsnelheid van uw batterij en laadpaal
- Het opwekvermogen van uw zonnepanelen
- Uw gemiddelde verbruik per uur tijdens het rijden
- De momenten waarop uw voertuig thuis beschikbaar is, in gebruik is of afwezig is

Op basis van deze gegevens simuleert de tool hoe u uw energie slim kunt verdelen. De tool regelt hoeveel energie u zelf opwekt, opslaat of teruglevert aan het net waardoor u inzichtelijk krijgt hoeveel u bespaart ten opzichte van een vast stroomtarief!
""")
# ─────────────────────────────── 1  Sidebar-input
with st.sidebar:
    st.header("🔧 Vul hieronder informatie in over uw systeem:")
    capacity     = st.number_input("Accucapaciteit (kWh)", 10.0, 200.0, 60.0)
    max_rate     = st.number_input("Max (ont)laad-snelheid (kW) van uw bidirectionele laadpaal", 1.0, 50.0, 12.8)
    soc_min_pct  = st.slider("Min SOC (%) hier mag uw batterijpercentage niet onder komen", 0, 100, 20)
    soc_max_pct  = st.slider("Max SOC (%) hier mag uw batterijpercentage niet boven komen", 0, 100, 80)
    efficiency   = st.slider("Efficiëntie export (%) round-trip efficiency. Standaard 85%", 50, 100, 85)/100
    drive_use    = st.number_input("Verbruik rijden (kWh/uur) hoeveel kWh gebruikt uw EV gemiddeld per uur rijden?", 0.0, 30.0, 10.0)
    st.header("☀️ PV & gegevens")
    pv_kwp = st.number_input("PV-vermogen (kWp) vul hier het vermogen van uw PV installatie in?", 0.0, 20.0, 4.0)
# ▼ Verbruiks­profiel: kies tabblad
if Path(USE_PATH).is_file():
    all_sheets = sheet_names(USE_PATH)           # haal actuele tab-namen op
    sheet_choice = st.selectbox(
        "Verbruiksprofiel (tabblad)",
        options = all_sheets,
        index   = 0,                             # eerste tab als default
        help    = "Kies het tabblad dat past bij uw type huishouden."
    )
else:
    st.error(f"❌ Verbruiksbestand '{USE_PATH}' niet gevonden"); st.stop()


# ─────────────────────────────── 3  Data-import

# bestanden direct inladen (géén uploads)
if Path(PRICE_PATH).is_file():
    price_df = load_prices(PRICE_PATH)
else:
    st.error(f"❌ Prijsbestand '{PRICE_PATH}' niet gevonden"); st.stop()

hours = pd.date_range(price_df.index.min(), price_df.index.max(), freq="h")
price_df = price_df.reindex(hours).ffill().bfill()

# PV
if Path(PV_PATH).is_file():
    pv_df = load_profile(PV_PATH, sheet=0, col="Opwek").reindex(hours).ffill().bfill()
else:
    st.error(f"❌ PV-bestand '{PV_PATH}' niet gevonden"); st.stop()
price_df["pv_kwh"] = pv_df["opwek"] * pv_kwp

# Verbruik
if Path(USE_PATH).is_file():
    use_df = load_profile(USE_PATH, sheet_choice, "Verbruik")   # gebruik keuze uit sidebar
else:
    st.error(f"❌ Verbruiksbestand '{USE_PATH}' niet gevonden"); st.stop()
price_df["verbruik"] = use_df.reindex(hours).ffill().bfill()["verbruik"]
df_base = price_df.copy()  # ← deze regel was verdwenen
if df_base.isnull().any().any():
    st.error("❌ Data bevat lege waarden"); st.stop()

# ─────────────────────────────── 4  Week-planner-UI (ongew.)
st.header("📆 Weekplanning – beschikbaarheid")
st.markdown("""
**Hoe vult u de weekplanning in?**

* Per dag kunt u aangeven op welke uren van de dag de auto beschikbaar/niet beschikbaar/in gebruik is. De simulator gebruikt dit profiel vervolgens voor het hele jaar.
* Kies per uur één van drie statussen:
  * **Beschikbaar**  de auto staat thuis; laden of ontladen is toegestaan.
  * **Niet beschikbaar**  de auto is fysiek weg; 
  * **In gebruik**  de auto is onderweg; het opgegeven *rij-verbruik (kWh/uur)* wordt uit de accu gehaald.

Handige knoppen:
* **Werkdag 08-17** vult de uren 08:00-17:00 met “Niet beschikbaar” (typische kantoordag).
* **Reset \<dag\>** herstelt één dag naar volledig “Beschikbaar”.
* **Reset hele week** zet alle dagen terug op “Beschikbaar”.
""")
opts=["Beschikbaar","Niet beschikbaar","In gebruik"]
colmap={"Beschikbaar":"#b6e2a1","Niet beschikbaar":"#f5a1a1","In gebruik":"#a1c8f5"}
if st.button("🔄 Reset hele week"):
    for d in range(7):
        for h in range(24): st.session_state[f"s_{d}_{h}"]="Beschikbaar"
for d,day in enumerate(["Maandag","Dinsdag","Woensdag","Donderdag","Vrijdag","Zaterdag","Zondag"]):
    st.subheader(day); r1,r2=st.columns(2)
    if r1.button(f"Reset {day}", key=f"rst{d}"):
        for h in range(24): st.session_state[f"s_{d}_{h}"]="Beschikbaar"
    if r2.button(f"Werkdag 08-17", key=f"wrk{d}"):
        for h in range(8,18): st.session_state[f"s_{d}_{h}"]="Niet beschikbaar"
    cols=st.columns(6)
    for h in range(24):
        k=f"s_{d}_{h}"; st.session_state.setdefault(k,"Beschikbaar")
        s=cols[h%6].selectbox(f"{h:02d}:00", opts, index=opts.index(st.session_state[k]), key=k)
        cols[h%6].markdown(f"<div style='background-color:{colmap[s]};height:18px;border-radius:3px'></div>",unsafe_allow_html=True)
def flags(idx):
    a,d=[],[]
    for ts in idx:
        stt=st.session_state.get(f"s_{ts.weekday()}_{ts.hour}","Beschikbaar")
        a.append(stt=="Beschikbaar"); d.append(stt=="In gebruik")
    return np.array(a),np.array(d)
avail_flags, drive_flags = flags(df_base.index)
soc_min,soc_max = soc_min_pct/100, soc_max_pct/100

# ─────────────────────────────── 5  Forecast-helpers
def make_forecast(idx,real,window=168):
    s=pd.Series(real,index=idx)
    fc=s.rolling(window,1).mean().shift(1)
    if fc.isna().any(): fc=fc.fillna(s.groupby(s.index.hour).transform("mean"))
    return fc.to_numpy()
def known_price_end(ts):
    return ts.normalize()+pd.Timedelta(hours=23) if ts.hour<12 else ts.normalize()+pd.Timedelta(days=1,hours=23)

# ─────────────────────────────── 6  MILP
def optimise(slice_df,soc0,avail,drive,load_fc,pv_fc):
    T=len(slice_df); inv=1/capacity; BIG=1_000
    buy=slice_df["Inkoop"].values; sell=slice_df["Verkoop"].values
    m=pulp.LpProblem("v2g", pulp.LpMaximize)
    c=dh=dg=imp=exp=soc=y=slk=None
    c  =pulp.LpVariable.dicts("c",range(T),0,max_rate)
    dh =pulp.LpVariable.dicts("dh",range(T),0,max_rate)
    dg =pulp.LpVariable.dicts("dg",range(T),0,max_rate)
    imp=pulp.LpVariable.dicts("imp",range(T),0)
    exp=pulp.LpVariable.dicts("exp",range(T),0)
    soc=pulp.LpVariable.dicts("soc",range(T),lowBound=soc_min,upBound=soc_max)
    y  =pulp.LpVariable.dicts("y",range(T),0,1,cat="Binary")
    slk=pulp.LpVariable.dicts("slk",range(T),0)
    m += pulp.lpSum(sell[t]*exp[t]-buy[t]*imp[t]-BIG*slk[t] for t in range(T))
    for t in range(T):
        m += imp[t]+pv_fc[t]+dh[t]+dg[t]==load_fc[t]+c[t]+exp[t]
        m += exp[t]==dg[t]*efficiency
        prev=soc0 if t==0 else soc[t-1]
        pen=drive_use if drive[t] else 0
        m += soc[t]==prev+(c[t]-dh[t]-dg[t]-pen)*inv+slk[t]*inv
        if not avail[t]: m += c[t]==0; m += dh[t]==0; m += dg[t]==0
        m += c[t]<=max_rate*y[t]; m += dh[t]+dg[t]<=max_rate*(1-y[t])
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    col=lambda v:[v[t].value() for t in range(T)]
    return pd.DataFrame({
        "charge":col(c),"discharge_home":col(dh),"discharge_grid":col(dg),
        "import":col(imp),"export":col(exp),"soc":col(soc)
    },index=slice_df.index)

# ─────────────────────────────── 7  MPC
def simulate(df_in):
    avg_b=df_in.groupby(df_in.index.hour)["Inkoop"].mean()
    avg_s=df_in.groupby(df_in.index.hour)["Verkoop"].mean()
    res={k:[] for k in ["charge","discharge_home","discharge_grid","import","export","soc"]}
    i,n,soc0=0,len(df_in),soc_min
    while i<n:
        hr=df_in.index[i].hour
        nxt=min([h for h in [0,6,12,18] if h>hr]+[24])
        commit,horizon=min(nxt-hr,n-i),min(nxt-hr+48,n-i)
        sel=df_in.iloc[i:i+horizon].copy()
        known=sel.index<=known_price_end(df_in.index[i])
        sel.loc[~known,"Inkoop"]=[avg_b[h] for h in sel.index[~known].hour]
        sel.loc[~known,"Verkoop"]=[avg_s[h] for h in sel.index[~known].hour]
        lf,pf=make_forecast(sel.index,sel["verbruik"]),make_forecast(sel.index,sel["pv_kwh"])
        out=optimise(sel,soc0,avail_flags[i:i+horizon],drive_flags[i:i+horizon],lf,pf)
        for k in range(commit):
            for col in res: res[col].append(out[col].iloc[k])
        soc0=out["soc"].iloc[commit-1]; i+=commit
    for col,arr in res.items(): df_in[col]=arr
    return df_in

# ─────────────────────────────── 8  Run-knop / cache
# 8  Run-knop / cache  – vervang dit blok
if st.button("⚡ Bereken uw situatie"):
    ui_sig = (capacity, max_rate, soc_min_pct, soc_max_pct,
              efficiency, drive_use, pv_kwp)         # ← nieuw
    prof_hash = hash(tuple(avail_flags)) ^ hash(tuple(drive_flags))
    @st.cache_data
    def simulate_cached(df_in, code_v, prof_h, ui_sig):  # ← extra arg
        return simulate(df_in.copy())
    st.session_state["sim_df"] = simulate_cached(
        df_base, CODE_VERSION, prof_hash, ui_sig
    )
# ─────────────────────────────── 9  Resultaten
if "sim_df" not in st.session_state:
    st.info("Klik op ⚡ Bereken strategie"); st.stop()
df = st.session_state["sim_df"]
if "pv_to_batt" not in df.columns:
    pv_to_house=np.minimum(df["pv_kwh"],df["verbruik"])
    pv_surplus =df["pv_kwh"]-pv_to_house
    df["pv_to_batt"]=np.minimum(pv_surplus,df["charge"])
    df["grid_to_batt"]=df["charge"]-df["pv_to_batt"]

# ▸ Kosten-KPI: drie scenario’s
TARIEF_FLAT = 0.30          # € / kWh vast
CYCLE_COST  = 0.03          # € / kWh ontladen, alleen voor V2G-scenario

# 0. uur-grootheden
df["drive_kWh"] = drive_use * drive_flags        # rit-verbruik
df["house_kWh"] = df["verbruik"]                 # huishoud-verbruik

# 1️⃣  Vast tarief, géén PV-saldering (auto altijd tot 100 %)
kwh_flat = df["house_kWh"] + df["drive_kWh"]
cost_flat = kwh_flat.sum() * TARIEF_FLAT

# 2️⃣  Vast tarief + PV met saldering
net_kwh = (df["house_kWh"] + df["drive_kWh"] - df["pv_kwh"]).sum()
net_kwh = max(net_kwh, 0)                     # nooit lager dan 0 kWh
cost_flat_pv = net_kwh * TARIEF_FLAT

# 3️⃣  Dynamisch + V2G + PV  (output uit optimiser)
cycle_penalty = CYCLE_COST * (
    (-df["discharge_home"] - df["discharge_grid"])
    .clip(lower=0)
).sum()

cost_dyn = (
      (df["import"] * df["Inkoop"]).sum()
    - (df["export"] * df["Verkoop"]).sum()
    + cycle_penalty
)

# ▸ KPI-weergave
col1, col2, col3, col4 = st.columns(4)
col1.metric("1. Vast tarief, 0,30€ (geen PV)",      f"€ {cost_flat:,.2f}")
col2.metric("2. Vast tarief, 0,30€ + PV (salderen)",f"€ {cost_flat_pv:,.2f}")
col3.metric("3. Dynamisch contract + V2G + PV",      f"€ {cost_dyn:,.2f}")

# relatieve besparing tov scenario 1
saving_abs = cost_flat - cost_dyn
saving_pct = saving_abs / cost_flat * 100 if cost_flat else 0
col4.metric("Besparing (%))", f"{saving_pct:.1f} %")


# ▸ Dag-aggregatie
st.subheader("📅 Dagelijkse resultaten: onderstaande tabel laat per dag zien wat er is gebeurd")
day = (df.resample("D")
          .agg({"verbruik":"sum","pv_kwh":"sum","pv_to_batt":"sum",
                "grid_to_batt":"sum","discharge_home":"sum",
                "discharge_grid":"sum","import":"sum","export":"sum"})
          .rename(columns={"verbruik":"verbruik_kWh","pv_kwh":"pv_kWh",
                "pv_to_batt":"laden_pv_kWh","grid_to_batt":"laden_net_kWh",
                "discharge_home":"ontladen_house_kWh",
                "discharge_grid":"ontladen_grid_kWh",
                "import":"import_kWh","export":"export_kWh"}))
st.dataframe(day,use_container_width=True)

# ▸ Uurtabel (eerste 168 uur = 1 week)
st.subheader("📋 Uur-detail: onderstaande tabel laat per uur zien wat er is gebeurd")
detail_cols=["verbruik","pv_kwh","pv_to_batt","grid_to_batt",
             "discharge_home","discharge_grid","import","export","soc"]
detail=df.iloc[:len(df)][detail_cols].copy()
detail=detail.rename(columns={
    "verbruik":"Verbruik (kWh)",
    "pv_kwh":"PV (kWh)",
    "pv_to_batt":"Laden PV→Accu",
    "grid_to_batt":"Laden Net→Accu",
    "discharge_home":"Ontl → Huis",
    "discharge_grid":"Ontl → Net",
    "import":"Import",
    "export":"Export",
    "soc":"SOC"
})
st.dataframe(detail,use_container_width=True)

# ▸ Interactieve daggrafiek
st.subheader("📈 Dagselectie: kies de gewenste dag en de grafiek laat zien wat er die dag is gebeurd")
days=df.index.normalize().unique()
chosen=st.date_input("Kies dag",value=days[0].date(),
                     min_value=days.min().date(),max_value=days.max().date())
dsel=df.loc[str(chosen)]

fig,ax_p=plt.subplots(figsize=(12,4))
ax_p.plot(dsel.index,dsel["Inkoop"],label="Inkoopprijs",color="tab:blue")
ax_p.plot(dsel.index,dsel["Verkoop"],label="Verkoopprijs",color="tab:cyan")
ax_p.set_ylabel("Prijs (€/kWh)"); ax_p.tick_params(axis="x",rotation=45)

ax_e=ax_p.twinx()
ax_e.bar(dsel.index,dsel["pv_to_batt"],width=0.03,label="Laden PV→Accu",color="forestgreen")
ax_e.bar(dsel.index,dsel["grid_to_batt"],width=0.03,bottom=dsel["pv_to_batt"],label="Laden Net→Accu",color="limegreen")
ax_e.bar(dsel.index,-dsel["discharge_home"],width=0.03,label="Ontl → Huis",color="orange")
ax_e.bar(dsel.index,-dsel["discharge_grid"],width=0.03,label="Ontl → Net",color="red",alpha=0.6)
ax_e.bar(dsel.index,dsel["import"],width=0.02,label="Import Net→Huis",color="grey",alpha=0.25)
ax_e.bar(dsel.index,-dsel["export"],width=0.02,label="Export Accu→Net",color="black",alpha=0.25)
ax_e.plot(dsel.index,dsel["soc"]*100,label="SOC (%)",color="tab:orange",linewidth=2)
ax_e.set_ylabel("Energie (kWh) / SOC (%)")

h1,l1=ax_p.get_legend_handles_labels()
h2,l2=ax_e.get_legend_handles_labels()
ax_p.legend(h1+h2,l1+l2,loc="upper left",ncol=2)
plt.tight_layout()
st.pyplot(fig)
