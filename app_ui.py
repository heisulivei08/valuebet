"""
Value bets: ottelulista -> simulaatio -> kokoonpanot -> pelaajan tilastot + markkinat (oma tn, kerroin, EV).
Datalähteet: API-Football (ottelut, tulokset, kokoonpanot, pelaajat, loukkaantumiset, tuomari),
football-data.co.uk (tuomarien kortti- ja rikekeskiarvot, vain kotiliigat), Valioliigan Fantasy-rajapinta (pelaajien xG/xA),
xG Stat (pelaajien xG/xA muissa liigoissa, vain kirjautumatta näkyvä osa). Understat on poistettu.
Avain: Streamlit Secrets -> API_FOOTBALL_KEY
requirements.txt: streamlit>=1.37, pandas, numpy, requests, beautifulsoup4
"""
import difflib
import re
import time
import unicodedata
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

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
USE_XGSTAT = True  # False = ei haeta xG Statista (Valioliigan Fantasy-rajapinta ja arviot jäävät käyttöön)
LEAGUES = {"Valioliiga": dict(af=39, fd="E0", xs=None), "La Liga": dict(af=140, fd="SP1", xs="la-liga"),
           "Serie A": dict(af=135, fd="I1", xs="serie-a"), "Bundesliiga": dict(af=78, fd="D1", xs="bundesliga"),
           "Ligue 1": dict(af=61, fd="F1", xs="ligue-1"), "Mestarien liiga": dict(af=2, fd=None, xs=None)}
TOP5 = ["Valioliiga", "La Liga", "Serie A", "Bundesliiga", "Ligue 1"]
UA_BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0 Safari/537.36"}
UA_APP = {"User-Agent": "Mozilla/5.0 (compatible; ValueBetsApp/1.0; personal use)"}
ALIAS = {  # API-Football -> football-data.co.uk
    "manchester city": "man city", "manchester united": "man united", "wolverhampton wanderers": "wolves",
    "nottingham forest": "nott'm forest", "newcastle united": "newcastle", "tottenham hotspur": "tottenham",
    "west ham united": "west ham", "brighton and hove albion": "brighton", "leicester city": "leicester",
    "leeds united": "leeds",
    "athletic club": "ath bilbao", "atletico madrid": "ath madrid", "real betis": "betis", "celta vigo": "celta",
    "espanyol": "espanol", "real sociedad": "sociedad", "rayo vallecano": "vallecano", "deportivo alaves": "alaves",
    "real oviedo": "oviedo", "real valladolid": "valladolid",
    "ac milan": "milan", "as roma": "roma", "hellas verona": "verona",
    "bayern munchen": "bayern munich", "borussia dortmund": "dortmund", "eintracht frankfurt": "ein frankfurt",
    "bayer leverkusen": "leverkusen", "borussia monchengladbach": "m'gladbach", "fsv mainz 05": "mainz",
    "1. fc koln": "fc koln", "vfb stuttgart": "stuttgart", "1899 hoffenheim": "hoffenheim", "fc st. pauli": "st pauli",
    "sc freiburg": "freiburg", "1. fc union berlin": "union berlin", "fc augsburg": "augsburg",
    "1. fc heidenheim": "heidenheim", "vfl wolfsburg": "wolfsburg",
    "paris saint germain": "paris sg", "olympique lyonnais": "lyon", "olympique de marseille": "marseille",
    "as monaco": "monaco", "stade rennais": "rennes", "rc strasbourg": "strasbourg", "stade brestois 29": "brest"}
POS = {"Goalkeeper": "GK", "Defender": "DF", "Midfielder": "MF", "Attacker": "FW"}


# ---------- apurit ----------
def asc(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def pkey(name):
    t = re.sub(r"[^a-z ]", " ", asc(name)).split()
    return (t[0][0], t[-1]) if t else ("", "")


def match_team(name, cands):
    raw = asc(name).strip()
    short = raw.replace(" fc", "").strip()
    cl = {asc(c): c for c in cands}
    for n in (ALIAS.get(raw), raw, ALIAS.get(short), short):
        if n and n in cl:
            return cl[n]
    m = difflib.get_close_matches(ALIAS.get(raw, short), list(cl), 1, 0.6)
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
    """Pelaajatilastot. league=None (Mestarien liiga): summataan kaikki kilpailut, jotta kotiliigan ottelut tulevat mukaan."""
    rows, page = [], 1
    while True:
        p = dict(team=team_id, season=season, page=page)
        if league:
            p["league"] = league
        j = api("players", **p)
        for it in j.get("response", []):
            st_ = it["statistics"]

            def tot(a, b):
                return sum(((x.get(a) or {}).get(b) or 0) for x in st_)

            rows.append(dict(af_id=it["player"]["id"], name=it["player"]["name"],
                             pos=POS.get((st_[0]["games"] or {}).get("position"), "?") if st_ else "?",
                             minutes=tot("games", "minutes"), apps=tot("games", "appearences") or tot("games", "appearances"),
                             starts=tot("games", "lineups"),
                             rating=(st_[0]["games"] or {}).get("rating") if st_ else None,
                             goals=tot("goals", "total"), assists=tot("goals", "assists"),
                             shots=tot("shots", "total"), sot=tot("shots", "on"), key_passes=tot("passes", "key"),
                             tackles=tot("tackles", "total"), fouls=tot("fouls", "committed"), drawn=tot("fouls", "drawn"),
                             yellow=tot("cards", "yellow"), red=tot("cards", "red")))
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


@st.cache_data(ttl=3600)
def af_team_matches(league, season):
    """Kotiliigat: liigan pelatut ottelut yhdellä pyynnöllä. Maalit ovat joukkueen voimaluvun pohja."""
    rows = []
    for f in api("fixtures", league=league, season=season, status="FT").get("response", []):
        gh, ga, d = f["goals"]["home"], f["goals"]["away"], f["fixture"]["date"][:10]
        if gh is None or ga is None:
            continue
        rows.append(dict(team=f["teams"]["home"]["name"], xG=float(gh), xGA=float(ga), date=d, h_a="h"))
        rows.append(dict(team=f["teams"]["away"]["name"], xG=float(ga), xGA=float(gh), date=d, h_a="a"))
    return pd.DataFrame(rows, columns=["team", "xG", "xGA", "date", "h_a"]).sort_values("date")


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


# ---------- Mestarien liiga: vastustajan taso huomioiva rating ----------
@st.cache_data(ttl=24 * 3600)
def team_matches(team_id):
    rows = []
    for f in api("fixtures", team=team_id, last=20, status="FT").get("response", []):
        gh, ga = f["goals"]["home"], f["goals"]["away"]
        if gh is None or ga is None:
            continue
        rows.append(dict(fid=f["fixture"]["id"], date=f["fixture"]["date"][:10], comp=f["league"]["name"],
                         home_id=f["teams"]["home"]["id"], away_id=f["teams"]["away"]["id"], hg=gh, ag=ga))
    return pd.DataFrame(rows, columns=["fid", "date", "comp", "home_id", "away_id", "hg", "ag"])


def fit(m, iters=80, ridge=2.0):
    """Painotettu Poisson: log(lambda) = mu + hyökkäys[tekijä] - puolustus[vastustaja] + kotietu. Ridge vetää kohti keskiarvoa."""
    teams = pd.unique(pd.concat([m.home_id, m.away_id]))
    idx = {t: i for i, t in enumerate(teams)}
    hi, ai = m.home_id.map(idx).to_numpy(), m.away_id.map(idx).to_numpy()
    hg, ag, w = m.hg.to_numpy(float), m.ag.to_numpy(float), m.w.to_numpy(float)
    n = len(teams)
    att, dfn = np.zeros(n), np.zeros(n)
    mu, h = np.log((hg.sum() + ag.sum()) / (2 * len(m))), 0.1
    for _ in range(iters):
        lh, la = np.exp(mu + att[hi] - dfn[ai] + h), np.exp(mu + att[ai] - dfn[hi])
        mu += (w * (hg - lh)).sum() / (w * lh).sum() * 0.5 + (w * (ag - la)).sum() / (w * la).sum() * 0.5
        h += (w * (hg - lh)).sum() / (w * lh).sum() * 0.5
        lh, la = np.exp(mu + att[hi] - dfn[ai] + h), np.exp(mu + att[ai] - dfn[hi])
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


@st.cache_data(ttl=6 * 3600, show_spinner="Haetaan joukkueiden ja vastustajien otteluhistoriaa (ensimmäisellä kerralla ~1 min)...")
def ratings_for(home_id, away_id):
    base = [team_matches(home_id), team_matches(away_id)]
    opp = set()
    for b in base:
        opp |= set(b.home_id) | set(b.away_id)
    m = pd.concat(base + [team_matches(int(t)) for t in sorted(opp - {home_id, away_id})]).drop_duplicates("fid")
    if len(m) < 10:
        return None
    age = (pd.Timestamp(date.today()) - pd.to_datetime(m.date)).dt.days.clip(lower=0)
    m["w"] = 0.5 ** (age / 240) * np.where(m.comp.str.lower().str.contains("friend"), 0.4, 1.0)
    return fit(m.reset_index(drop=True))


def rat_gd(r, tid):
    """Odotettu maaliero per ottelu keskitasoista vastustajaa vastaan."""
    return np.exp(r["mu"] + r["att"].get(tid, 0.0)) - np.exp(r["mu"] - r["dfn"].get(tid, 0.0))


# ---------- pelaajien xG: Valioliiga = Fantasy-rajapinta, muut = xG Stat (vain kirjautumatta näkyvä osa) ----------
@st.cache_data(ttl=6 * 3600)
def fpl_players():
    """Valioliiga: virallinen Fantasy-rajapinta (ilmainen, ei avainta). Kauden xG ja xA."""
    d = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", headers=UA_BROWSER, timeout=30).json()
    els = d["elements"]
    if not els or "expected_goals" not in els[0]:
        raise RuntimeError("FPL-rajapinnassa ei ole xG-kenttiä")
    df = pd.DataFrame([dict(name=f"{e['first_name']} {e['second_name']}", mins=float(e["minutes"]),
                            xG=float(e.get("expected_goals") or 0), xA=float(e.get("expected_assists") or 0)) for e in els])
    df["k"] = df.name.map(pkey)
    return df


def parse_xgstat(html):
    """Lukee xG Statin pelaajataulukon: nimi tulee pelaajalinkin osoitteesta (/players/erling-haaland), luvut otsikoiden mukaan."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    head = None
    for t in tables:
        ths = [th.get_text(" ", strip=True) for th in t.find_all("th")]
        if "Name" in ths and "xG" in ths and "Mins" in ths:
            head = ths
            break
    if not head:
        return []
    idx = {h: i for i, h in enumerate(head)}
    num = lambda s: float(re.sub(r"[^0-9.\-]", "", s.replace("−", "-")) or 0)
    out = []
    for t in tables:
        for tr in t.find_all("tr"):
            tds = tr.find_all("td")
            a = tr.find("a", href=re.compile(r"/players/[^/?#]+$"))
            if len(tds) != len(head) or not a:
                continue
            name = a["href"].rstrip("/").split("/")[-1].replace("-", " ")
            c = [td.get_text(" ", strip=True) for td in tds]
            out.append(dict(name=name, mins=num(c[idx["Mins"]]), xG=num(c[idx["xG"]]),
                            xA=num(c[idx["xA"]]) if "xA" in idx else 0.0))
    return out


@st.cache_data(ttl=12 * 3600, show_spinner="Haetaan pelaajien xG-tietoja...")
def xgstat_players(slug, max_pages=3):
    """Hakee korkeintaan 3 sivua (ilmainen osa), 2 s välein, ilman kirjautumista. 429 tai tyhjä sivu pysäyttää haun."""
    rows = []
    for p in range(1, max_pages + 1):
        url = f"https://www.xgstat.com/competitions/{slug}/players" + (f"?page={p}" if p > 1 else "")
        r = requests.get(url, headers=UA_APP, timeout=30)
        got = parse_xgstat(r.text) if r.status_code == 200 else []
        if not got:
            if p == 1:
                raise RuntimeError(f"xG Stat ei palauttanut taulukkoa (HTTP {r.status_code})")
            break
        rows += got
        time.sleep(2.0)
    df = pd.DataFrame(rows).drop_duplicates("name")
    df["k"] = df.name.map(pkey)
    return df


def xg_table(league_name):
    """Yhdistää lähteet: Valioliiga = FPL, muut viisi = xG Stat. Mestarien liigassa kaikki viisi (pelaajan kotiliiga)."""
    names = TOP5 if league_name == "Mestarien liiga" else [league_name]
    frames, errs = [], []
    for n in names:
        try:
            if n == "Valioliiga":
                frames.append(fpl_players())
            elif USE_XGSTAT and LEAGUES.get(n, {}).get("xs"):
                fails = st.session_state.setdefault("xg_fail", {})
                if time.time() - fails.get(n, 0) < 1800:  # viime yritys epäonnistui: ei uutta pyyntöä 30 min
                    errs.append(f"{n}: odotetaan 30 min viime virheen jälkeen")
                    continue
                try:
                    frames.append(xgstat_players(LEAGUES[n]["xs"]))
                except Exception:
                    fails[n] = time.time()
                    raise
        except Exception as e:
            errs.append(f"{n}: {type(e).__name__} {str(e)[:60]}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["name", "mins", "xG", "xA", "k"])
    return df, errs


def player_odds(fx):
    """KYTKE TÄHÄN KERROINLÄHDE. Palauta {(pkey(pelaaja), markkina): kerroin}, esim. {(("e","haaland"), "Maali 1+"): 1.85}.
    Markkinat: 'Maali 1+', 'Syöttö 1+', 'Kortti 1+', 'Rike 1+', 'Rike 2+'."""
    return {}


# ---------- laskenta ----------
def ref_factors(fd, ref, k=8):
    if fd is None or not ref:
        return 1.0, 1.0, 0
    lc, lf = fd.cards.mean(), fd.fouls.mean()
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
    oppid = fx["away_id" if side == "home" else "home_id"]
    if ctx["rat"] is not None:  # Mestarien liiga: vastustajan puolustus ratingistä
        f_xga, f_draw = float(np.exp(-ctx["rat"]["dfn"].get(oppid, 0.0))), 1.0
    else:
        tm = ctx["teams"]
        oppu = match_team(opp, tm.team.unique())
        if not oppu:
            st.warning(f"Joukkueen nimeä ei voitu yhdistää: {name} / {opp}")
            return pd.DataFrame()
        oppf = match_team(opp, set(ctx["fd"].HomeTeam) | set(ctx["fd"].AwayTeam))
        if not oppf:
            st.caption(f"Tuomari- ja rikedatasta ei löytynyt joukkuetta {opp}: rikekerroin 1.00.")
        f_xga = tm[tm.team == oppu].tail(10).xGA.mean() / tm.xG.mean()
        f_draw = drawn_factor(ctx["fd"], oppf) if oppf else 1.0
    if f_xga != f_xga:
        f_xga = 1.0
    m = af_players(None if ctx["rat"] is not None else cfg["af"], season, tid)
    if m.empty:
        return m
    m = m.drop_duplicates("af_id").copy()
    m["k"] = m.name.map(pkey)
    # varalla xG- ja xA-arvio: puolet toteutuneista, puolet laukauksista (0.30 maalia / maalia kohti -laukaus) ja avainsyötöistä (0.12)
    m["xG"] = 0.5 * m.goals + 0.5 * m.sot * 0.30
    m["xA"] = 0.5 * m.assists + 0.5 * m.key_passes * 0.12
    m["tmin"] = m.minutes.astype(float)
    m["xsrc"] = "arvio"
    xt = ctx.get("xgt")
    if xt is not None and len(xt):
        by_k = {k: g for k, g in xt.groupby("k")}
        for i, r in m.iterrows():
            g = by_k.get(r.k)
            if g is None:
                continue
            j = (g.mins - r.minutes).abs().idxmin()  # useampi samanniminen: lähimmät minuutit
            src = g.loc[j]
            if src.mins <= r.minutes * 1.2 + 90:  # lähteen liigaminuutit eivät voi ylittää kaikkia minuutteja
                m.loc[i, ["xG", "xA", "tmin"]] = [src.xG, src.xA, src.mins]
                m.loc[i, "xsrc"] = "FPL/xG Stat"
    m["status"] = m.af_id.map(lambda i: "Poissa" if inj.get(i) == "Missing Fixture" else "Epävarma" if i in inj else "")
    n90 = (m.minutes / 90).clip(lower=3)
    mpa = (m.minutes / m.apps.clip(lower=1)).clip(20, 90)
    em = np.where(m.af_id.isin(confirmed), np.maximum(mpa, 70), mpa)
    t90 = m.tmin.clip(lower=270) / 90
    m["xg90"] = m.xG / t90
    m["lam_g"] = m.xg90 * em / 90 * f_xga
    m["lam_f"] = m.fouls / n90 * em / 90 * f_draw * ref_f
    m["lam_c"] = (m.yellow + m.red) / n90 * em / 90 * ref_c * f_draw ** 0.5
    m["lam_a"] = m.xA / t90 * em / 90 * f_xga
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

    def vs(a, b):
        return f"{a} vs {b:.1f} ({a - b:+.1f})"

    st.dataframe(pd.DataFrame({
        "Tilasto": ["Ottelut / aloitukset", "Minuutit", "Arvosana", "Maalit / syötöt", "xG / per 90 (lähde)",
                    "Laukaukset / maalia kohti", "Avainsyötöt", "Taklaukset", "Rikkeet tehty / saatu",
                    "Keltaiset / punaiset", "Odotetut minuutit"],
        "Arvo": [f"{r['apps']} / {r['starts']}", str(r["minutes"]), str(r["rating"] or "-"),
                 f"{r['goals']} / {r['assists']}", f"{r['xG']:.2f} / {r['xg90']:.2f} ({r['xsrc']})",
                 f"{r['shots']} / {r['sot']}", str(r["key_passes"]), str(r["tackles"]),
                 f"{r['fouls']} / {r['drawn']}", f"{r['yellow']} / {r['red']}", f"{r['exp_min']:.0f}"]}),
        hide_index=True, use_container_width=True)
    st.markdown("**Viimeistely ja ennuste tähän otteluun**")
    st.dataframe(pd.DataFrame({
        "Mittari": ["Maalit vs xG-arvio", "Syötöt vs xA-arvio", "Ennustettu maaliodotus tässä ottelussa",
                    "Ennustettu syöttöodotus tässä ottelussa"],
        "Arvo": [vs(r["goals"], r["xG"]), vs(r["assists"], r["xA"]), f"{r['lam_g']:.2f}", f"{r['lam_a']:.2f}"]}),
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
    """Kotiliigat: koti-/vierassuhde (viim. 10) + kauden keskiarvo, vastustajan puolustus huomioiden."""
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


def goals_table(tm, home, away):
    rows = []
    for name in (home, away):
        d = tm[tm.team == match_team(name, tm.team.unique())]
        for n in (5, 10):
            x = d.tail(n)
            rows.append({"Joukkue": name, "Otos": f"Viim. {n}", "Tehdyt": x.xG.mean(), "Päästetyt": x.xGA.mean()})
    return pd.DataFrame(rows)


# ---------- UI ----------
st.title("Value bets")
if not KEY:
    st.error("API_FOOTBALL_KEY puuttuu: lisää se Streamlitin Settings → Secrets.")
    st.stop()

league = st.selectbox("Liiga", list(LEAGUES), label_visibility="collapsed")
cfg = LEAGUES[league]
is_cl = cfg["fd"] is None
today = date.today()
season = str(today.year if today.month >= 7 else today.year - 1)

if "fx" not in st.session_state:
    fxs = af_fixtures(cfg["af"], season, today.isoformat(), (today + timedelta(days=7)).isoformat())
    if not fxs:
        st.info("Ei otteluita seuraavan 7 päivän aikana.")
    for f in fxs:
        if st.button(f"{f['date']}  ·  {f['home']} – {f['away']}", key=f"m{f['id']}", use_container_width=True):
            st.session_state.fx = f
            st.session_state.is_cl = is_cl
            st.session_state.lg = league
            st.rerun()
    st.stop()

fx, is_cl = st.session_state.fx, st.session_state.get("is_cl", is_cl)
league = st.session_state.get("lg", league)
cfg = LEAGUES[league]
if st.button("← Takaisin"):
    del st.session_state["fx"]
    st.rerun()

try:
    if is_cl:
        rat = ratings_for(fx["home_id"], fx["away_id"])
        if rat is None:
            st.error("Otteluhistoriaa ei löytynyt tarpeeksi mallin sovittamiseen.")
            st.stop()
        ctx = dict(fd=None, teams=None, rat=rat)
        lh = float(np.exp(rat["mu"] + rat["att"].get(fx["home_id"], 0) - rat["dfn"].get(fx["away_id"], 0) + rat["h"]))
        la = float(np.exp(rat["mu"] + rat["att"].get(fx["away_id"], 0) - rat["dfn"].get(fx["home_id"], 0)))
    else:
        ctx = dict(fd=load_fd(cfg["fd"], season), teams=af_team_matches(cfg["af"], season), rat=None)
        if ctx["teams"].empty:
            st.error("Liigan otteludataa ei löytynyt kaudelle " + season)
            st.stop()
        lh, la = team_lams(ctx["teams"], fx["home"], fx["away"])
except Exception as e:
    st.error(f"Datan haku epäonnistui: {type(e).__name__}: {e}")
    st.stop()

ctx["xgt"], xerr = xg_table(league)
for e in xerr:
    st.caption(f"xG-lähde ei vastannut ({e}): käytetään arviota.")
ref_c, ref_f, n_ref = ref_factors(ctx["fd"], fx["ref"])
inj, lineups, odds = af_injuries(fx["id"]), af_lineups(fx["id"]), player_odds(fx)

st.header(f"{fx['home']} – {fx['away']}")
meta = f"{fx['date']} · Tuomari: {fx['ref'] or 'ei vielä julkaistu'}"
meta += f" · korttikerroin {ref_c:.2f} (otos {n_ref})" if not is_cl else " · tuomaridataa ei Mestarien liigassa (kerroin 1.00)"
st.markdown(f"<div class='meta'>{meta}</div>", unsafe_allow_html=True)
if is_cl:
    st.caption("Mestarien liigassa joukkueiden voimaluvut sovitetaan koko otteluverkosta (kotiliigat + eurokisat), "
               "joten vastustajan taso huomioidaan. Pelaajien xG ja xA haetaan pelaajan kotiliigasta (viisi suurta liigaa); "
               "muut pelaajat saavat arvion.")
else:
    st.caption("Maaliodotukset perustuvat liigan toteutuneisiin maaleihin (koti-/vierassuhde). Pelaajien xG ja xA tulevat "
               "Valioliigassa Fantasy-rajapinnasta ja muissa liigoissa xG Statista; puuttuvilla pelaajilla käytetään arviota.")

if lh == lh and la == la:
    with st.expander(f"Simulaatio (20 000 ottelua) · odotetut maalit {lh:.2f} – {la:.2f}", expanded=True):
        show_sim(sim_cached(lh, la, ref_c), fx["home"], fx["away"])

if is_cl:
    with st.expander("Joukkueiden vahvuus (vastustaja huomioitu)"):
        rows = [{"Joukkue": n, "Hyökkäys": np.exp(ctx["rat"]["mu"] + ctx["rat"]["att"].get(t, 0)),
                 "Päästetyt": np.exp(ctx["rat"]["mu"] - ctx["rat"]["dfn"].get(t, 0)), "Maaliero": rat_gd(ctx["rat"], t),
                 "Otteluita": ctx["rat"]["n"].get(t, 0)} for n, t in ((fx["home"], fx["home_id"]), (fx["away"], fx["away_id"]))]
        st.dataframe(pd.DataFrame(rows).style.format({"Hyökkäys": "{:.2f}", "Päästetyt": "{:.2f}", "Maaliero": "{:+.2f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Hyökkäys = maalia/ottelu keskitasoista puolustusta vastaan. Päästetyt = maalia/ottelu keskitasoista hyökkäystä "
                   "vastaan (pienempi on parempi). Maaliero = odotettu maaliero keskitasoista vastustajaa vastaan.")
else:
    with st.expander("Joukkueiden maalit (viim. 5 ja 10)"):
        st.dataframe(goals_table(ctx["teams"], fx["home"], fx["away"]).style.format(
            {"Tehdyt": "{:.2f}", "Päästetyt": "{:.2f}"}, na_rep="-"), hide_index=True, use_container_width=True)

for side in ("home", "away"):
    conf = lineups.get(fx[f"{side}_id"], [])
    m = squad(cfg, season, fx, side, ctx, ref_c, ref_f, inj, conf)
    if m.empty:
        continue
    xi, label = pick_xi(m, conf)
    st.subheader(fx[side])
    st.caption(f"{label} · oikea xG {int((m.xsrc != 'arvio').sum())}/{len(m)} pelaajalla, muilla arvio")
    for _, r in xi.sort_values("pos", key=lambda s: s.map({"GK": 0, "DF": 1, "MF": 2, "FW": 3})).iterrows():
        pm = markets(r)
        tag = f" ⚠ {r['status']}" if r["status"] else ""
        txt = f"{r['pos']}  {r['name']}{tag}  ·  maali {pm['Maali 1+']:.0%}  ·  kortti {pm['Kortti 1+']:.0%}"
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
