"""
Value bets: ottelulista -> kokoonpanot -> pelaajan tilastot + markkinat (oma tn, kerroin, EV).
Datalähteet: API-Football (kokoonpanot, pelaajat, loukkaantumiset, tuomari), Understat (xG),
football-data.co.uk (tuomarit, aiheutetut rikkeet). Avain: Streamlit Secrets -> API_FOOTBALL_KEY
requirements.txt: streamlit>=1.37, pandas, numpy, requests, understatapi
"""
import difflib
import re
import unicodedata
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
try:  # understatapi kutsuu selenium-metodia set_headless, joka on poistettu uusista selenium-versioista
    from selenium.webdriver.common.options import ArgOptions

    if not hasattr(ArgOptions, "set_headless"):
        ArgOptions.set_headless = lambda self, headless=True: None
except Exception:
    pass
from understatapi import UnderstatClient

from simulation import simulate

st.set_page_config(page_title="Value bets", page_icon="📈", layout="centered")
st.markdown("""<style>
#MainMenu, footer {visibility:hidden;}
.block-container {padding:1rem 1rem 4rem; max-width:560px;}
h1 {font-size:1.5rem !important; font-weight:700;}
.meta {color:#7d8a99; font-size:.8rem; margin:-6px 0 8px;}
div.stButton > button {border-radius:12px; text-align:left; justify-content:flex-start; border:1px solid #1f2a36; background:#131a22;}
div.stButton > button:hover {border-color:#22c55e; color:#22c55e;}
</style>""", unsafe_allow_html=True)

try:
    KEY = st.secrets["API_FOOTBALL_KEY"]
except Exception:
    KEY = ""
BASE = "https://v3.football.api-sports.io"
LEAGUES = {"Valioliiga": dict(us="EPL", af=39, fd="E0"), "La Liga": dict(us="La_liga", af=140, fd="SP1"),
           "Serie A": dict(us="Serie_A", af=135, fd="I1"), "Bundesliiga": dict(us="Bundesliga", af=78, fd="D1"),
           "Ligue 1": dict(us="Ligue_1", af=61, fd="F1")}
ALIAS = {"manchester city": "man city", "manchester united": "man united", "wolverhampton wanderers": "wolves",
         "nottingham forest": "nott'm forest", "newcastle united": "newcastle", "tottenham hotspur": "tottenham",
         "west ham united": "west ham", "brighton and hove albion": "brighton", "leicester city": "leicester",
         "leeds united": "leeds"}
POS = {"Goalkeeper": "GK", "Defender": "DF", "Midfielder": "MF", "Attacker": "FW"}


# ---------- apurit ----------
def asc(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def pkey(name):
    t = re.sub(r"[^a-z ]", " ", asc(name)).split()
    return (t[0][0], t[-1]) if t else ("", "")


def match_team(name, cands):
    n = asc(name).replace(" fc", "").strip()
    n = ALIAS.get(n, n)
    cl = {asc(c): c for c in cands}
    if n in cl:
        return cl[n]
    m = difflib.get_close_matches(n, list(cl), 1, 0.6)
    return cl[m[0]] if m else None


def api(path, **params):
    j = requests.get(f"{BASE}/{path}", headers={"x-apisports-key": KEY}, params=params, timeout=30).json()
    if j.get("errors"):
        st.error(f"API-Football ({path}): {j['errors']}")
    return j


# ---------- data ----------
@st.cache_data(ttl=3600)
def af_fixtures(league, season, d0, d1):
    j = api("fixtures", league=league, season=season, **{"from": d0, "to": d1})
    return [dict(id=f["fixture"]["id"], date=f["fixture"]["date"][:16].replace("T", " "), ref=f["fixture"].get("referee"),
                 home=f["teams"]["home"]["name"], home_id=f["teams"]["home"]["id"],
                 away=f["teams"]["away"]["name"], away_id=f["teams"]["away"]["id"]) for f in j.get("response", [])]


@st.cache_data(ttl=24 * 3600)
def af_players(league, season, team_id):
    rows, page = [], 1
    while True:
        j = api("players", league=league, season=season, team=team_id, page=page)
        for it in j.get("response", []):
            s = it["statistics"][0]
            g, sh, go, pa, ta = s["games"], s["shots"], s["goals"], s["passes"], s["tackles"]
            rows.append(dict(af_id=it["player"]["id"], name=it["player"]["name"], pos=POS.get(g.get("position"), "?"),
                             minutes=g.get("minutes") or 0, apps=g.get("appearences") or g.get("appearances") or 0,
                             starts=g.get("lineups") or 0, rating=g.get("rating"),
                             goals=go.get("total") or 0, assists=go.get("assists") or 0,
                             shots=sh.get("total") or 0, sot=sh.get("on") or 0, key_passes=pa.get("key") or 0,
                             tackles=ta.get("total") or 0, fouls=s["fouls"]["committed"] or 0,
                             drawn=s["fouls"]["drawn"] or 0, yellow=s["cards"]["yellow"] or 0, red=s["cards"]["red"] or 0))
        if page >= j.get("paging", {}).get("total", 1):
            break
        page += 1
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def af_lineups(fixture_id):
    j = api("fixtures/lineups", fixture=fixture_id)
    return {t["team"]["id"]: [p["player"]["id"] for p in t["startXI"]] for t in j.get("response", [])}


@st.cache_data(ttl=1800)
def af_injuries(fixture_id):
    j = api("injuries", fixture=fixture_id)
    return {r["player"]["id"]: r["player"].get("type", "") for r in j.get("response", [])}


@st.cache_data(ttl=6 * 3600)
def us_players(code, season):
    with UnderstatClient() as us:
        p = pd.DataFrame(us.league(league=code).get_player_data(season=season))
    cols = ["time", "xG", "xA", "xGChain", "xGBuildup"]
    p[cols] = p[cols].apply(pd.to_numeric)
    p["k"] = p.player_name.map(pkey)
    return p


@st.cache_data(ttl=3600)
def us_teams(code, season):
    with UnderstatClient() as us:
        d = us.league(league=code).get_team_data(season=season)
    return pd.DataFrame([dict(team=t["title"], xG=float(m["xG"]), xGA=float(m["xGA"]), date=m["date"], h_a=m["h_a"],
                              deep=float(m["deep"]), deep_allowed=float(m["deep_allowed"]))
                         for t in d.values() for m in t["history"]]).sort_values("date")


@st.cache_data(ttl=6 * 3600)
def load_fd(code, season):
    yy, fr = int(season) % 100, []
    for y in (yy, yy - 1):
        try:
            fr.append(pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{y:02d}{(y + 1) % 100:02d}/{code}.csv"))
        except Exception:
            pass
    d = pd.concat(fr).dropna(subset=["HomeTeam"])
    d["cards"], d["fouls"] = d.HY + d.AY + d.HR + d.AR, d.HF + d.AF
    return d


def player_odds(fx):
    """KYTKE TÄHÄN KERROINLÄHDE. Palauta {(pkey(pelaaja), markkina): kerroin}, esim. {(("e","haaland"), "Maali 1+"): 1.85}.
    Markkinat: 'Maali 1+', 'Kortti 1+', 'Rike 1+', 'Rike 2+'."""
    return {}


# ---------- laskenta ----------
def ref_factors(fd, ref, k=8):
    lc, lf = fd.cards.mean(), fd.fouls.mean()
    if not ref:
        return 1.0, 1.0, 0
    key = pkey(ref.split(",")[0])
    sub = fd[[pkey(x) == key for x in fd.Referee.fillna("")]]
    n = len(sub)
    return ((sub.cards.sum() + k * lc) / (n + k)) / lc, ((sub.fouls.sum() + k * lf) / (n + k)) / lf, n


def drawn_factor(fd, t):
    d = pd.concat([fd[fd.HomeTeam == t].AF, fd[fd.AwayTeam == t].HF]).mean()
    return 1.0 if np.isnan(d) else d / ((fd.HF.mean() + fd.AF.mean()) / 2)


def squad(cfg, season, fx, side, ctx, ref_c, ref_f, inj, confirmed):
    name, tid = fx[side], fx[f"{side}_id"]
    opp = fx["away" if side == "home" else "home"]
    tm = ctx["teams"]
    ust, oppu = match_team(name, tm.team.unique()), match_team(opp, tm.team.unique())
    oppf = match_team(opp, set(ctx["fd"].HomeTeam) | set(ctx["fd"].AwayTeam))
    if not (ust and oppu and oppf):
        st.warning(f"Joukkueen nimeä ei voitu yhdistää: {name} / {opp}")
        return pd.DataFrame()
    f_xga = tm[tm.team == oppu].tail(10).xGA.mean() / tm.xG.mean()
    f_draw = drawn_factor(ctx["fd"], oppf)
    af = af_players(cfg["af"], season, tid)
    if af.empty:
        return af
    af["k"] = af.name.map(pkey)
    us_ = ctx["usp"][ctx["usp"].team_title.str.contains(ust, regex=False)]
    m = af.merge(us_[["k", "xG", "time", "xA", "xGChain", "xGBuildup"]], on="k", how="left").drop_duplicates("af_id")
    m["status"] = m.af_id.map(lambda i: "Poissa" if inj.get(i) == "Missing Fixture" else "Epävarma" if i in inj else "")
    n90 = (m.minutes / 90).clip(lower=3)
    mpa = (m.minutes / m.apps.clip(lower=1)).clip(20, 90)
    em = np.where(m.af_id.isin(confirmed), np.maximum(mpa, 70), mpa)
    m["xg90"] = m.xG / (m.time.fillna(m.minutes).clip(lower=270) / 90)
    m["lam_g"] = m.xg90 * em / 90 * f_xga
    m["lam_f"] = m.fouls / n90 * em / 90 * f_draw * ref_f
    m["lam_c"] = (m.yellow + m.red) / n90 * em / 90 * ref_c * f_draw ** 0.5
    m["lam_a"] = m.xA / (m.time.fillna(m.minutes).clip(lower=270) / 90) * em / 90 * f_xga
    m["exp_min"] = em
    return m


def pick_xi(m, confirmed):
    if confirmed:
        return m[m.af_id.isin(confirmed)], "Vahvistettu kokoonpano"
    a = m[m.status != "Poissa"].sort_values(["starts", "minutes"], ascending=False)
    xi = pd.concat([a[a.pos == p].head(k) for p, k in {"GK": 1, "DF": 4, "MF": 4, "FW": 2}.items()])
    return xi, "Arvioitu kokoonpano (4-4-2, eniten aloituksia)"


def markets(r):
    g, c, f = r["lam_g"], r["lam_c"], r["lam_f"]
    return {"Maali 1+": 1 - np.exp(-g), "Syöttö 1+": 1 - np.exp(-r["lam_a"]), "Kortti 1+": 1 - np.exp(-c),
            "Rike 1+": 1 - np.exp(-f), "Rike 2+": 1 - np.exp(-f) * (1 + f)}


@st.dialog("Pelaaja")
def player_dialog(r, odds):
    st.subheader(f"{r['name']} · {r['pos']}")
    if r["status"]:
        st.warning(f"Tila: {r['status']}")
    xg = f"{r['xG']:.2f} / {r['xg90']:.2f}" if r["xG"] == r["xG"] else "-"
    st.dataframe(pd.DataFrame({"Tilasto": ["Ottelut / aloitukset", "Minuutit", "Arvosana", "Maalit / syötöt", "xG (kausi) / xG/90",
                                           "Laukaukset / maalia kohti", "Avainsyötöt", "Taklaukset", "Rikkeet tehty / saatu",
                                           "Keltaiset / punaiset", "Odotetut minuutit"],
                               "Arvo": [f"{r['apps']} / {r['starts']}", str(r["minutes"]), str(r["rating"] or "-"),
                                        f"{r['goals']} / {r['assists']}", xg,
                                        f"{r['shots']} / {r['sot']}", str(r["key_passes"]), str(r["tackles"]),
                                        f"{r['fouls']} / {r['drawn']}", f"{r['yellow']} / {r['red']}", f"{r['exp_min']:.0f}"]}),
                 hide_index=True, use_container_width=True)
    def n2(v):
        return f"{v:.2f}" if v == v else "-"

    def vs(a, b):
        return f"{a} vs {b:.1f} ({a - b:+.1f})" if b == b else "-"

    st.markdown("**Viimeistely ja ennuste tähän otteluun**")
    st.dataframe(pd.DataFrame({
        "Mittari": ["Maalit vs xG", "Syötöt vs xA", "xGChain / xGBuildup", "Ennustettu xG tässä ottelussa",
                    "Ennustettu xA tässä ottelussa"],
        "Arvo": [vs(r["goals"], r["xG"]), vs(r["assists"], r["xA"]),
                 f"{n2(r['xGChain'])} / {n2(r['xGBuildup'])}", n2(r["lam_g"]), n2(r["lam_a"])]}),
        hide_index=True, use_container_width=True)
    rows = []
    for mk, p in markets(r).items():
        o = odds.get((r["k"], mk))
        rows.append({"Markkina": mk, "Oma tn": p, "Oikea kerroin": 1 / p if p and p > 0 else np.nan,
                     "Kirjan kerroin": o, "EV": p * o - 1 if o and p == p else np.nan})
    st.caption("Oma todennäköisyys huomioi vastustajan, tuomarin ja odotetut minuutit.")
    st.dataframe(pd.DataFrame(rows).style.format({"Oma tn": "{:.0%}", "Oikea kerroin": "{:.2f}", "Kirjan kerroin": "{:.2f}",
                                                  "EV": "{:+.1%}"}, na_rep="-")
                 .map(lambda v: "background-color:#14532d;color:#22c55e" if isinstance(v, float) and v > 0 else "", subset=["EV"]),
                 hide_index=True, use_container_width=True)


def team_lams(tm, home, away):
    """Joukkueiden odotetut maalit: koti-/vierassuhde (viim. 10) + kauden keskiarvo, vastustajan puolustus huomioiden."""
    h, a = match_team(home, tm.team.unique()), match_team(away, tm.team.unique())
    lg = tm.xG.mean()

    def b(team, col, side):
        d = tm[tm.team == team]
        return 0.6 * d[d.h_a == side].tail(10)[col].mean() + 0.4 * d[col].mean()

    return b(h, "xG", "h") * b(a, "xGA", "a") / lg, b(a, "xG", "a") * b(h, "xGA", "h") / lg


@st.cache_data(ttl=3600)
def sim_cached(lh, la, rc):
    return simulate(lh, la, rc)


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


def threat_table(tm, home, away):
    """xT-vastine: xG, xGA ja 'deep' (rangaistusalueen lähelle päätyneet syötöt/kuljetukset) per ottelu."""
    rows = []
    for name in (home, away):
        d = tm[tm.team == match_team(name, tm.team.unique())]
        for n in (5, 10):
            x = d.tail(n)
            rows.append({"Joukkue": name, "Otos": f"Viim. {n}", "xG": x.xG.mean(), "xGA": x.xGA.mean(),
                         "Deep": x.deep.mean(), "Deep sallittu": x.deep_allowed.mean()})
    return pd.DataFrame(rows)


# ---------- UI ----------
st.title("Value bets")
if not KEY:
    st.error("API_FOOTBALL_KEY puuttuu: lisää se Streamlitin Settings → Secrets.")
    st.stop()

league = st.selectbox("Liiga", list(LEAGUES), label_visibility="collapsed")
cfg = LEAGUES[league]
today = date.today()
season = str(today.year if today.month >= 7 else today.year - 1)

if "fx" not in st.session_state:
    fxs = af_fixtures(cfg["af"], season, today.isoformat(), (today + timedelta(days=7)).isoformat())
    if not fxs:
        st.info("Ei otteluita seuraavan 7 päivän aikana.")
    for f in fxs:
        if st.button(f"{f['date']}  ·  {f['home']} – {f['away']}", key=f"m{f['id']}", use_container_width=True):
            st.session_state.fx = f
            st.rerun()
    st.stop()

fx = st.session_state.fx
if st.button("← Takaisin"):
    del st.session_state["fx"]
    st.rerun()

ctx = dict(fd=load_fd(cfg["fd"], season), teams=us_teams(cfg["us"], season), usp=us_players(cfg["us"], season))
ref_c, ref_f, n_ref = ref_factors(ctx["fd"], fx["ref"])
inj, lineups, odds = af_injuries(fx["id"]), af_lineups(fx["id"]), player_odds(fx)

st.header(f"{fx['home']} – {fx['away']}")
st.markdown(f"<div class='meta'>{fx['date']} · Tuomari: {fx['ref'] or 'ei vielä julkaistu'} "
            f"· korttikerroin {ref_c:.2f} (otos {n_ref})</div>", unsafe_allow_html=True)

lh, la = team_lams(ctx["teams"], fx["home"], fx["away"])
if lh == lh and la == la:
    with st.expander(f"Simulaatio (20 000 ottelua) · xG {lh:.2f} – {la:.2f}", expanded=True):
        show_sim(sim_cached(lh, la, ref_c), fx["home"], fx["away"])

with st.expander("Joukkueiden uhka (xT-vastine)"):
    st.dataframe(threat_table(ctx["teams"], fx["home"], fx["away"]).style.format(
        {c: "{:.2f}" for c in ["xG", "xGA", "Deep", "Deep sallittu"]}, na_rep="-"),
        hide_index=True, use_container_width=True)
    st.caption("Deep = rangaistusalueen lähelle (n. 20 jaardia maalista) päätyneet syötöt ja kuljetukset per ottelu. "
               "Se on lähin ilmainen vastine xT:lle.")

for side in ("home", "away"):
    conf = lineups.get(fx[f"{side}_id"], [])
    m = squad(cfg, season, fx, side, ctx, ref_c, ref_f, inj, conf)
    if m.empty:
        continue
    xi, label = pick_xi(m, conf)
    st.subheader(fx[side])
    st.caption(label)
    for _, r in xi.sort_values("pos", key=lambda s: s.map({"GK": 0, "DF": 1, "MF": 2, "FW": 3})).iterrows():
        pm = markets(r)
        tag = f" ⚠ {r['status']}" if r["status"] else ""
        txt = (f"{r['pos']}  {r['name']}{tag}  ·  maali {pm['Maali 1+']:.0%}  ·  kortti {pm['Kortti 1+']:.0%}"
               if pm["Maali 1+"] == pm["Maali 1+"] else f"{r['pos']}  {r['name']}{tag}  ·  kortti {pm['Kortti 1+']:.0%}")
        if st.button(txt, key=f"{side}{r['af_id']}", use_container_width=True):
            player_dialog(r.to_dict(), odds)
    out = m[~m.af_id.isin(xi.af_id) & (m.status != "Poissa")]
    with st.expander(f"Muut pelaajat ({len(out)})"):
        for _, r in out.sort_values("minutes", ascending=False).iterrows():
            if st.button(f"{r['pos']}  {r['name']}", key=f"o{side}{r['af_id']}", use_container_width=True):
                player_dialog(r.to_dict(), odds)
    gone = m[m.status == "Poissa"].name.tolist()
    if gone:
        st.caption("Poissa: " + ", ".join(gone))
