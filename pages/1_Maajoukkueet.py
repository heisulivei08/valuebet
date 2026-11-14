"""
Maajoukkuesivu (erillinen lisäsivu): sijoita repoon pages/1_Maajoukkueet.py
Realistinen malli: jokaiselle joukkueelle sovitetaan hyökkäys- ja puolustusluku koko otteluverkosta (Poisson-regressio),
joten vastustajan taso huomioidaan. Verkko = molempien joukkueiden viim. 20 ottelua + näiden vastustajien viim. 20 ottelua.
Data: API-Football. Avain: Streamlit Secrets -> API_FOOTBALL_KEY
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from simulation import simulate

st.set_page_config(page_title="Maajoukkueet", page_icon="🌍", layout="centered")
st.markdown("""<style>
.block-container {padding:1rem 1rem 4rem; max-width:560px;}
div.stButton > button {border-radius:12px; text-align:left; justify-content:flex-start; border:1px solid #1f2a36; background:#131a22;}
div.stButton > button:hover {border-color:#22c55e; color:#22c55e;}
</style>""", unsafe_allow_html=True)

try:
    KEY = st.secrets["API_FOOTBALL_KEY"]
except Exception:
    KEY = ""
BASE = "https://v3.football.api-sports.io"
EXCL = ("women", "u17", "u18", "u19", "u20", "u21", "u23", "club", "youth", "olympic")
TOURNAMENTS = ("world cup", "euro", "copa america", "africa cup", "asian cup", "gold cup", "nations cup")
POS = {"Goalkeeper": "GK", "Defender": "DF", "Midfielder": "MF", "Attacker": "FW"}


def api(path, **params):
    j = requests.get(f"{BASE}/{path}", headers={"x-apisports-key": KEY}, params=params, timeout=30).json()
    if j.get("errors"):
        st.error(f"API-Football ({path}): {j['errors']}")
    return j


def is_neutral(comp):
    c = comp.lower()
    return any(t in c for t in TOURNAMENTS) and "qualif" not in c


@st.cache_data(ttl=6 * 3600)
def national_fixtures(days):
    out = []
    for i in range(days):
        d = (date.today() + timedelta(days=i)).isoformat()
        for f in api("fixtures", date=d).get("response", []):
            lg = f["league"]
            if lg.get("country") == "World" and not any(x in lg["name"].lower() for x in EXCL):
                out.append(dict(id=f["fixture"]["id"], date=f["fixture"]["date"][:16].replace("T", " "),
                                league=lg["name"], ref=f["fixture"].get("referee"),
                                home=f["teams"]["home"]["name"], home_id=f["teams"]["home"]["id"],
                                away=f["teams"]["away"]["name"], away_id=f["teams"]["away"]["id"]))
    return out


@st.cache_data(ttl=24 * 3600)
def team_matches(team_id):
    rows = []
    for f in api("fixtures", team=team_id, last=20, status="FT").get("response", []):
        gh, ga = f["goals"]["home"], f["goals"]["away"]
        if gh is None or ga is None:
            continue
        rows.append(dict(fid=f["fixture"]["id"], date=f["fixture"]["date"][:10], comp=f["league"]["name"],
                         home_id=f["teams"]["home"]["id"], home=f["teams"]["home"]["name"],
                         away_id=f["teams"]["away"]["id"], away=f["teams"]["away"]["name"], hg=gh, ag=ga))
    return pd.DataFrame(rows, columns=["fid", "date", "comp", "home_id", "home", "away_id", "away", "hg", "ag"])


def fit(m, iters=80, ridge=2.0):
    """Painotettu Poisson: log(lambda) = mu + hyökkäys[tekijä] - puolustus[vastustaja] + koti*kotietu. Ridge vetää kohti keskiarvoa."""
    teams = pd.unique(pd.concat([m.home_id, m.away_id]))
    idx = {t: i for i, t in enumerate(teams)}
    hi, ai = m.home_id.map(idx).to_numpy(), m.away_id.map(idx).to_numpy()
    hg, ag, w, hf = m.hg.to_numpy(float), m.ag.to_numpy(float), m.w.to_numpy(float), (~m.neutral).to_numpy(float)
    n = len(teams)
    att, dfn = np.zeros(n), np.zeros(n)
    mu, h = np.log((hg.sum() + ag.sum()) / (2 * len(m))), 0.1
    for _ in range(iters):
        lh, la = np.exp(mu + att[hi] - dfn[ai] + h * hf), np.exp(mu + att[ai] - dfn[hi])
        mu += (w * (hg - lh)).sum() / (w * lh).sum() * 0.5 + (w * (ag - la)).sum() / (w * la).sum() * 0.5
        h += (w * (hg - lh) * hf).sum() / max((w * lh * hf).sum(), 1e-9) * 0.5
        lh, la = np.exp(mu + att[hi] - dfn[ai] + h * hf), np.exp(mu + att[ai] - dfn[hi])
        g_att = np.bincount(hi, w * (hg - lh), n) + np.bincount(ai, w * (ag - la), n) - ridge * att
        s_att = np.bincount(hi, w * lh, n) + np.bincount(ai, w * la, n) + ridge
        g_dfn = -(np.bincount(hi, w * (ag - la), n) + np.bincount(ai, w * (hg - lh), n)) - ridge * dfn
        s_dfn = np.bincount(hi, w * la, n) + np.bincount(ai, w * lh, n) + ridge
        att += 0.7 * g_att / s_att
        dfn += 0.7 * g_dfn / s_dfn
        ma, md = att.mean(), dfn.mean()
        att, dfn, mu = att - ma, dfn - md, mu + ma - md
    cnt = pd.concat([m.home_id, m.away_id]).value_counts()
    return dict(mu=mu, h=h, att=dict(zip(teams, att)), dfn=dict(zip(teams, dfn)), n=cnt.to_dict())


@st.cache_data(ttl=6 * 3600, show_spinner="Haetaan joukkueiden ja vastustajien otteluhistoriaa (kestää ensimmäisellä kerralla ~1 min)...")
def ratings_for(home_id, away_id):
    base = [team_matches(home_id), team_matches(away_id)]
    opp = set()
    for b in base:
        opp |= set(b.home_id) | set(b.away_id)
    frames = base + [team_matches(int(t)) for t in sorted(opp - {home_id, away_id})]
    m = pd.concat(frames).drop_duplicates("fid").reset_index(drop=True)
    if len(m) < 10:
        return None, m
    age = (pd.Timestamp(date.today()) - pd.to_datetime(m.date)).dt.days.clip(lower=0)
    m["w"] = 0.5 ** (age / 365) * np.where(m.comp.str.lower().str.contains("friend"), 0.7, 1.0)
    m["neutral"] = m.comp.map(is_neutral)
    return fit(m), m


def pred(r, hid, aid, neutral):
    g = lambda d, k: d.get(k, 0.0)
    lh = np.exp(r["mu"] + g(r["att"], hid) - g(r["dfn"], aid) + (0 if neutral else r["h"]))
    la = np.exp(r["mu"] + g(r["att"], aid) - g(r["dfn"], hid))
    return lh, la


def gd(r, tid):
    """Odotettu maaliero per ottelu keskitasoista vastustajaa vastaan neutraalilla kentällä."""
    return np.exp(r["mu"] + r["att"].get(tid, 0.0)) - np.exp(r["mu"] - r["dfn"].get(tid, 0.0))


@st.cache_data(ttl=24 * 3600)
def nat_players(team_id, season):
    rows, page = [], 1
    while True:
        j = api("players", team=team_id, season=season, page=page)
        for it in j.get("response", []):
            s = it["statistics"]
            rows.append(dict(af_id=it["player"]["id"], name=it["player"]["name"],
                             pos=POS.get(s[0]["games"].get("position"), "?") if s else "?",
                             minutes=sum(x["games"].get("minutes") or 0 for x in s),
                             apps=sum((x["games"].get("appearences") or x["games"].get("appearances") or 0) for x in s),
                             goals=sum(x["goals"].get("total") or 0 for x in s),
                             assists=sum(x["goals"].get("assists") or 0 for x in s),
                             yellow=sum(x["cards"].get("yellow") or 0 for x in s),
                             red=sum(x["cards"].get("red") or 0 for x in s)))
        if page >= j.get("paging", {}).get("total", 1):
            break
        page += 1
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def lineups(fid):
    return {t["team"]["id"]: [p["player"]["id"] for p in t["startXI"]]
            for t in api("fixtures/lineups", fixture=fid).get("response", [])}


@st.cache_data(ttl=1800)
def injuries(fid):
    return {r["player"]["id"]: r["player"].get("type", "") for r in api("injuries", fixture=fid).get("response", [])}


@st.cache_data(ttl=3600)
def sim_cached(lh, la):
    return simulate(lh, la, 1.0)


def player_table(pl, conf, inj, lam, base):
    if pl.empty:
        return pl
    p = pl[pl.minutes >= 90].copy()
    p = p[p.af_id.map(lambda i: inj.get(i) != "Missing Fixture")]
    n90 = (p.minutes / 90).clip(lower=3)
    mpa = (p.minutes / p.apps.clip(lower=1)).clip(20, 90)
    em = np.where(p.af_id.isin(conf), np.maximum(mpa, 70), mpa)
    f = lam / base  # joukkueen odotetut maalit tässä ottelussa vs. sen keskitaso
    p["Pelaaja"], p["Pos"] = p.name, p.pos
    p["Maali 1+ %"] = (1 - np.exp(-p.goals / n90 * em / 90 * f)) * 100
    p["Syöttö 1+ %"] = (1 - np.exp(-p.assists / n90 * em / 90 * f)) * 100
    p["Kortti 1+ %"] = (1 - np.exp(-(p.yellow + p.red) / n90 * em / 90)) * 100
    p["Aloittaa"] = np.where(p.af_id.isin(conf), "✓", "")
    p["Tila"] = p.af_id.map(lambda i: "Epävarma" if i in inj else "")
    return p[["Pelaaja", "Pos", "Aloittaa", "Tila", "Maali 1+ %", "Syöttö 1+ %", "Kortti 1+ %"]]


def show_sim(r, home, away):
    p = r["p"]
    c1, c2, c3 = st.columns(3)
    c1.metric(home, f"{p[0]:.0%}", f"oik. kerroin {1 / p[0]:.2f}", delta_color="off")
    c2.metric("Tasapeli", f"{p[1]:.0%}", f"{1 / p[1]:.2f}", delta_color="off")
    c3.metric(away, f"{p[2]:.0%}", f"{1 / p[2]:.2f}", delta_color="off")
    st.caption(f"Odotetut maalit {r['goals'][0]:.2f} – {r['goals'][1]:.2f} · molemmat tekevät {r['btts']:.0%} · "
               f"punainen kortti {r['red']:.0%} · ensimmäinen maali: koti {r['first'][0]:.0%}, vieras {r['first'][1]:.0%}, "
               f"ei maalia {r['first'][2]:.0%}")
    st.dataframe(pd.DataFrame({"Markkina": list(r["ou"]), "Tn": list(r["ou"].values()),
                               "Oikea kerroin": [1 / v for v in r["ou"].values()]})
                 .style.format({"Tn": "{:.0%}", "Oikea kerroin": "{:.2f}"}), hide_index=True, use_container_width=True)
    st.markdown("**Todennäköisimmät tulokset**")
    st.bar_chart(r["scores"])
    st.markdown("**Maalit jaksoittain**")
    st.bar_chart(r["timing"])
    st.markdown("**Pelinkulku: kuka johtaa ottelun aikana (minuutti)**")
    st.line_chart(r["lead"])


# ---------- UI ----------
st.title("Maajoukkueet")
if not KEY:
    st.error("API_FOOTBALL_KEY puuttuu: lisää se Streamlitin Settings → Secrets.")
    st.stop()

if "nfx" not in st.session_state:
    fxs = national_fixtures(7)
    if not fxs:
        st.info("Ei maajoukkueotteluita seuraavan 7 päivän aikana (tai API palautti virheen).")
    last = None
    for f in sorted(fxs, key=lambda x: (x["league"], x["date"])):
        if f["league"] != last:
            st.subheader(f["league"])
            last = f["league"]
        if st.button(f"{f['date']}  ·  {f['home']} – {f['away']}", key=f"n{f['id']}", use_container_width=True):
            st.session_state.nfx = f
            st.rerun()
    st.stop()

fx = st.session_state.nfx
if st.button("← Takaisin"):
    del st.session_state["nfx"]
    st.rerun()

rat, mm = ratings_for(fx["home_id"], fx["away_id"])
st.header(f"{fx['home']} – {fx['away']}")
if rat is None:
    st.error("Otteluhistoriaa ei löytynyt tarpeeksi mallin sovittamiseen.")
    st.stop()

neutral = is_neutral(fx["league"])
lh, la = pred(rat, fx["home_id"], fx["away_id"], neutral)
st.caption(f"{fx['date']} · {fx['league']} · tuomari: {fx['ref'] or 'ei vielä julkaistu'} · "
           f"{'neutraali kenttä' if neutral else f'kotietu mukana (x{np.exp(rat['h']):.2f})'}")
st.caption(f"Malli sovitettiin {len(mm)} otteluun ja {len(rat['att'])} joukkueeseen. Jokaisen joukkueen luvut on korjattu "
           "sen vastustajien tasolla, ja vanhat ottelut painavat vähemmän (puoliintumisaika 1 vuosi, ystävyysottelut x0.7).")

with st.expander(f"Simulaatio (20 000 ottelua) · odotetut maalit {lh:.2f} – {la:.2f}", expanded=True):
    show_sim(sim_cached(lh, la), fx["home"], fx["away"])

with st.expander("Joukkueiden vahvuus (vastustaja huomioitu)"):
    rows = []
    for name, tid in ((fx["home"], fx["home_id"]), (fx["away"], fx["away_id"])):
        rows.append({"Joukkue": name, "Hyökkäys": np.exp(rat["mu"] + rat["att"].get(tid, 0)),
                     "Päästetyt": np.exp(rat["mu"] - rat["dfn"].get(tid, 0)), "Maaliero": gd(rat, tid),
                     "Otteluita": rat["n"].get(tid, 0)})
    st.dataframe(pd.DataFrame(rows).style.format({"Hyökkäys": "{:.2f}", "Päästetyt": "{:.2f}", "Maaliero": "{:+.2f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Hyökkäys = maalia/ottelu keskitasoista puolustusta vastaan. Päästetyt = maalia/ottelu keskitasoista hyökkäystä "
               "vastaan (pienempi on parempi). Maaliero = odotettu maaliero keskitasoista vastustajaa vastaan neutraalilla kentällä.")
    for name, tid in ((fx["home"], fx["home_id"]), (fx["away"], fx["away_id"])):
        t = mm[(mm.home_id == tid) | (mm.away_id == tid)].sort_values("date", ascending=False).head(10)
        out = pd.DataFrame({
            "Pvm": t.date,
            "Vastustaja": np.where(t.home_id == tid, t.away, t.home),
            "Tulos": np.where(t.home_id == tid, t.hg.astype(str) + "–" + t.ag.astype(str),
                              t.ag.astype(str) + "–" + t.hg.astype(str)),
            "Vastustajan maaliero": [gd(rat, int(o)) for o in np.where(t.home_id == tid, t.away_id, t.home_id)]})
        st.markdown(f"**{name}: viimeiset 10 ottelua ja vastustajan taso**")
        st.dataframe(out.style.format({"Vastustajan maaliero": "{:+.2f}"}), hide_index=True, use_container_width=True)

conf, inj = lineups(fx["id"]), injuries(fx["id"])
season = str(date.today().year if date.today().month >= 7 else date.today().year - 1)
for name, tid, lam in ((fx["home"], fx["home_id"], lh), (fx["away"], fx["away_id"], la)):
    st.subheader(name)
    c = conf.get(tid, [])
    st.caption("Vahvistettu kokoonpano" if c else "Kokoonpano ei vielä julkaistu: näytetään kaikki pelaajat")
    base = np.exp(rat["mu"] + rat["att"].get(tid, 0))
    t = player_table(nat_players(tid, season), c, inj, lam, base)
    if t.empty:
        st.info("Pelaajatilastoja ei löytynyt tälle joukkueelle.")
        continue
    t = t[t.Aloittaa == "✓"] if c else t
    st.dataframe(t.sort_values("Maali 1+ %", ascending=False), hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.ProgressColumn(k.replace(" %", ""), format="%.0f%%", min_value=0, max_value=100)
                                for k in ["Maali 1+ %", "Syöttö 1+ %", "Kortti 1+ %"]})
