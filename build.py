# -*- coding: utf-8 -*-
"""
여론조사 종합 추세 빌더
- 입력: polls.csv (중앙선거여론조사심의위원회 등록 항목 기준 1행 = 1개 조사)
- 출력: index.html (JS 없이도 전부 보이는 단일 파일, 차트는 빌드 시 SVG로 생성)
실행: python build.py [polls.csv] [index.html]
"""
import sys, math, base64, html, datetime as dt
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "polls.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "index.html"
FONT = "num.woff2"

# ---------------------------------------------------------------- 모형 설정
SIGMA = 7.0          # 추세 커널 폭(일)
CUT = 21             # 커널 반영 범위(일)
N_CAP = 3000         # 표본 가중 상한
METHOD_W = {"전화면접": 1.0, "무선ARS": 0.9, "혼합": 0.95, "웹": 0.8}
SHRINK = 2.0         # 기관 효과 축소(가상 조사 2건만큼 0 쪽으로)
FREQ_WIN = 10        # 같은 기관 조사 밀집도 계산 범위(일)

PARTIES = [("dem", "더불어민주당", "#1B5FAA"), ("ppp", "국민의힘", "#D6262F"),
           ("rebuild", "조국혁신당", "#0B2F6B"), ("reform", "개혁신당", "#E87511"),
           ("progressive", "진보당", "#9C1D3F")]
C_APP, C_DIS = "#1F7A5C", "#C6452B"
SHORT = {"NBS(엠브레인퍼블릭·케이스탯리서치·코리아리서치·한국리서치)": "NBS 전국지표조사",
         "한국사회여론연구소(KSOI)": "한국사회여론연구소"}


# ---------------------------------------------------------------- 서식 도우미
def hu(x, nd=1):
    q = Decimal(1).scaleb(-nd)
    return Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)

def f1(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{hu(x,1):.1f}"

def f0(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{int(hu(x,0)):,}"

def rd(a, b):
    """표시(반올림)값끼리의 차이: 화면 숫자와 차이값이 어긋나지 않게"""
    return float(hu(a, 1)) - float(hu(b, 1))

def sg(x):
    v = hu(x, 1)
    if v == 0:
        return "0.0"
    return f"+{v:.1f}" if v > 0 else f"-{abs(v):.1f}"

def esc(s):
    return html.escape(str(s), quote=True)

def kdate(d):
    return f"{d.month}월 {d.day}일"

def sdate(d):
    return f"{d.month}.{d.day}"


# ---------------------------------------------------------------- 자료 정리
df = pd.read_csv(SRC)
df["field_start"] = pd.to_datetime(df["field_start"])
df["field_end"] = pd.to_datetime(df["field_end"])
df["mid"] = df["field_start"] + (df["field_end"] - df["field_start"]) / 2
df["mid"] = df["mid"].dt.normalize()
df["org"] = df["pollster"].map(lambda s: SHORT.get(s, s))
df["unit"] = df["org"] + " · " + df["method"]
df["sponsor"] = df["sponsor"].fillna("")
df["sp_short"] = df["sponsor"].map(lambda s: "자체조사" if "자체" in s else (s or "미기재"))
df["sp_type"] = np.where(df["sponsor"] == "", "미기재", np.where(df["sponsor"].str.contains("자체"), "자체조사", "언론사 의뢰"))
df["mw"] = df["method"].map(METHOD_W).fillna(0.85)
df["nw"] = np.sqrt(np.minimum(df["sample_n"], N_CAP) / 1000.0)
D0 = df["mid"].min()
df["t"] = (df["mid"] - D0).dt.days.astype(float)
T_END = int(df["t"].max())
DAYS = np.arange(0, T_END + 1)
DATES = [D0 + pd.Timedelta(days=int(i)) for i in DAYS]
LAST_FIELD = df["field_end"].max()


def freq_adj(sub):
    out = []
    for _, r in sub.iterrows():
        same = sub[(sub["unit"] == r["unit"]) & ((sub["t"] - r["t"]).abs() <= FREQ_WIN)]
        out.append(1.0 / math.sqrt(len(same)))
    return np.array(out)


def fit(col):
    """기관 효과 보정 + 커널 가중 추세. 반환: 추세, 띠(상·하), 기관 효과, 보정값"""
    sub = df[df[col].notna()].copy()
    y = sub[col].to_numpy(float)
    t = sub["t"].to_numpy(float)
    units = sub["unit"].to_numpy()
    base_w = sub["nw"].to_numpy() * sub["mw"].to_numpy() * freq_adj(sub)
    house = {u: 0.0 for u in set(units)}

    def trend_at(tt, yy):
        k = np.exp(-0.5 * ((tt[:, None] - t[None, :]) / SIGMA) ** 2)
        k[np.abs(tt[:, None] - t[None, :]) > CUT] = 0
        w = k * base_w[None, :]
        s = w.sum(1)
        m = (w * yy[None, :]).sum(1) / np.where(s > 0, s, np.nan)
        return m, w, s

    for _ in range(8):
        adj = y - np.array([house[u] for u in units])
        m_pts, _, _ = trend_at(t, adj)
        resid = y - m_pts
        newh = {}
        for u in house:
            r = resid[units == u]
            newh[u] = r.sum() / (len(r) + SHRINK)
        cen = np.mean(list(newh.values()))
        house = {u: v - cen for u, v in newh.items()}

    adj = y - np.array([house[u] for u in units])
    m, w, s = trend_at(DAYS.astype(float), adj)
    m_pts, _, _ = trend_at(t, adj)
    r2 = (adj - m_pts) ** 2
    sd = np.sqrt((w * r2[None, :]).sum(1) / np.where(s > 0, s, np.nan))
    # 마지막 조사 이후는 표시하지 않음, 첫 조사 이전도 표시하지 않음
    lo_t, hi_t = t.min(), t.max()
    valid = (DAYS >= lo_t) & (DAYS <= hi_t)
    for i in DAYS:
        near = set(units[np.abs(t - i) <= 10])
        if len(near) < 2:
            valid[i] = False
    m[~valid] = np.nan
    sd[~valid] = np.nan
    sub["adj"] = adj
    return dict(trend=m, up=m + 1.28 * sd, lo=m - 1.28 * sd, house=house, sub=sub,
                n=len(sub), sd=sd)


MODELS = {c: fit(c) for c in ["approve", "disapprove"] + [p[0] for p in PARTIES] + ["none"]}


def at(col, date):
    i = (pd.Timestamp(date) - D0).days
    v = MODELS[col]["trend"]
    i = max(0, min(i, len(v) - 1))
    # 유효값이 없으면 가장 가까운 유효값
    if np.isnan(v[i]):
        idx = np.where(~np.isnan(v))[0]
        i = idx[np.argmin(np.abs(idx - i))]
    return float(v[i]), DATES[i]


def last(col):
    v = MODELS[col]["trend"]
    i = np.where(~np.isnan(v))[0].max()
    return float(v[i]), DATES[i]


def first(col):
    v = MODELS[col]["trend"]
    i = np.where(~np.isnan(v))[0].min()
    return float(v[i]), DATES[i]


# ---------------------------------------------------------------- SVG 차트
FONT_STACK = "'PretendardSub','Pretendard','Apple SD Gothic Neo','Noto Sans KR','Malgun Gothic',sans-serif"
NUMF = FONT_STACK  # Pretendard 숫자(민짜 0) + tabular-nums


def spread(items, lo, hi, gap):
    """라벨 y값 겹침 해소: items=[(y, key)] -> {key: y}"""
    items = sorted(items)
    ys = [y for y, _ in items]
    for _ in range(200):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                d = (gap - (ys[i] - ys[i - 1])) / 2
                ys[i - 1] -= d
                ys[i] += d
                moved = True
        ys = [min(max(y, lo), hi) for y in ys]
        if not moved:
            break
    return {k: y for (_, k), y in zip(items, ys)}


def line_chart(series, ymin, ymax, ystep, W, H, mobile=False, events=(), cols_points=True, marks=(), cross_at=None):
    """series: [{col, name, color}]"""
    fs = 11 if mobile else 12.5
    L, R, T, B = (30, 92, 14, 30) if mobile else ((36, 112, 16, 32) if W < 600 else (40, 140, 16, 32))
    rb = 8 if mobile else 9.5
    levels = []
    if marks:
        pw0 = W - L - R
        last_x = {0: -1e9, 1: -1e9}
        for dte, _ in marks:
            x = L + (pd.Timestamp(dte) - D0).days / T_END * pw0
            lv = 0 if x - last_x[0] > 2 * rb + 3 else 1
            last_x[lv] = x
            levels.append(lv)
        add = (2 * rb + 8) + (2 * rb + 3) * max(levels)
        T = add + 4
        H += add + 4 - (14 if mobile else 16)
    pw, ph = W - L - R, H - T - B
    x0 = 0
    x1 = T_END
    X = lambda t: L + (t - x0) / (x1 - x0) * pw
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * ph
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" style="display:block">']
    # 격자 + y 눈금
    v = ymin
    while v <= ymax + 1e-9:
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line2)" stroke-width="1" stroke-dasharray="1 4" stroke-linecap="round"/>')
        o.append(f'<text x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="{fs-1}" fill="var(--muted)" font-family="{NUMF}">{int(v)}</text>')
        v += ystep
    # x 눈금: 매월 1일
    for d in DATES:
        if d.day == 1:
            tt = (d - D0).days
            o.append(f'<line x1="{X(tt):.1f}" x2="{X(tt):.1f}" y1="{T+ph}" y2="{T+ph+5}" stroke="var(--muted)"/>')
            o.append(f'<text x="{X(tt):.1f}" y="{T+ph+18}" text-anchor="middle" font-size="{fs}" fill="var(--muted)">{d.month}월</text>')
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{T+ph}" y2="{T+ph}" stroke="var(--line2)"/>')
    # 사건 표시
    for (d1, d2, lab) in events:
        a, b = (pd.Timestamp(d1) - D0).days, (pd.Timestamp(d2) - D0).days
        if 0 <= a <= x1:
            o.append(f'<rect x="{X(a):.1f}" y="{T}" width="{max(2,X(min(b,x1))-X(a)):.1f}" height="{ph}" fill="var(--gold-s)"/>')
            o.append(f'<text x="{X(a)-4:.1f}" y="{T+ph-6:.1f}" text-anchor="end" font-size="{fs-1.5}" fill="var(--gold)" stroke="var(--bg)" stroke-width="3" paint-order="stroke">{lab}</text>')
    # 변곡점 사건: 점선 + 번호 배지(가까우면 두 줄로 엇갈림)
    for k, (dte, _txt) in enumerate(marks, 1):
        tt = (pd.Timestamp(dte) - D0).days
        if not (0 <= tt <= x1):
            continue
        x = X(tt)
        lv = levels[k - 1]
        by = T - (rb + 4) - lv * (2 * rb + 3)
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{by+rb:.1f}" y2="{T+ph}" stroke="var(--ev)" stroke-width="1" stroke-dasharray="3 3" stroke-opacity="0.7"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{by:.1f}" r="{rb}" fill="var(--ev)"/>')
        o.append(f'<text x="{x:.1f}" y="{by+rb*0.42:.1f}" text-anchor="middle" font-size="{rb*1.25:.1f}" font-weight="700" fill="#FFFFFF" font-family="{NUMF}">{k}</text>')
    # 두 계열 사이 면: 앞선 쪽 색으로 칠함(2계열일 때)
    if len(series) == 2:
        A, Bm = MODELS[series[0]["col"]]["trend"], MODELS[series[1]["col"]]["trend"]
        idx = [i for i in DAYS if not np.isnan(A[i]) and not np.isnan(Bm[i])]
        runs, cur = [], []
        for i in idx:
            sgn = A[i] >= Bm[i]
            if cur and (cur[-1][1] != sgn):
                runs.append(cur); cur = [cur[-1][0:1] + (sgn,)] if False else [(cur[-1][0], sgn)]
            cur.append((i, sgn))
        if cur:
            runs.append(cur)
        for run in runs:
            ii = [i for i, _ in run]
            sgn = run[-1][1]
            col = series[0]["color"] if sgn else series[1]["color"]
            top = " ".join(f"{X(i):.1f},{Y(A[i]):.1f}" for i in ii)
            bot = " ".join(f"{X(i):.1f},{Y(Bm[i]):.1f}" for i in reversed(ii))
            o.append(f'<polygon points="{top} {bot}" fill="{col}" fill-opacity="0.13" stroke="none"/>')
    # 개별 조사: 조사기간만큼 짧은 가로선(전화면접 굵게, ARS 가늘게)
    if cols_points:
        for s in series:
            sub = MODELS[s["col"]]["sub"]
            for _, r in sub.iterrows():
                t1 = max(0, (r["field_start"] - D0).days)
                t2 = min(x1, (r["field_end"] - D0).days)
                xa, xb = X(t1), X(max(t2, t1))
                if xb - xa < 3:
                    xa, xb = (xa + xb) / 2 - 1.5, (xa + xb) / 2 + 1.5
                yv = Y(min(max(r[s["col"]], ymin), ymax))
                tip = esc(f'{r["org"]} · {r["method"]} · {sdate(r["field_start"])}~{sdate(r["field_end"])} · n={int(r["sample_n"]):,} · {s["name"]} {f1(r[s["col"]])}%')
                if r["method"] == "전화면접":
                    sw, op = (2.6 if mobile else 3.2), 0.55
                else:
                    sw, op = (1.4 if mobile else 1.7), 0.5
                o.append(f'<line x1="{xa:.1f}" x2="{xb:.1f}" y1="{yv:.1f}" y2="{yv:.1f}" stroke="{s["color"]}" stroke-opacity="{op}" stroke-width="{sw}" stroke-linecap="round"><title>{tip}</title></line>')
    # 추세선
    for s in series:
        M = MODELS[s["col"]]
        pts = [f"{X(i):.1f},{Y(M['trend'][i]):.1f}" for i in DAYS if not np.isnan(M["trend"][i])]
        o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="var(--bg)" stroke-width="{5.5 if mobile else 6.5}" stroke-linejoin="round" stroke-linecap="round"/>')
        o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{s["color"]}" stroke-width="{2.4 if mobile else 2.8}" stroke-linejoin="round" stroke-linecap="round"/>')
    # 교차 시점 표시
    if cross_at is not None:
        tt = (cross_at - D0).days
        v = float(MODELS[series[0]["col"]]["trend"][tt])
        cx, cy = X(tt), Y(v)
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{4.5 if not mobile else 3.8}" fill="var(--bg)" stroke="var(--ink)" stroke-width="2"/>')
        o.append(f'<text x="{cx:.1f}" y="{cy+(24 if not mobile else 34):.1f}" text-anchor="middle" font-size="{fs}" font-weight="700" fill="var(--ink)" stroke="var(--bg)" stroke-width="4" paint-order="stroke"><tspan font-family="{NUMF}">{cross_at.month}.{cross_at.day}</tspan> 역전</text>')
    # 끝값 라벨(겹침 해소)
    ends = []
    for s in series:
        v, d = last(s["col"])
        ends.append((Y(v), s["col"]))
    pos = spread(ends, T + 10, T + ph - 8, fs * (3.2 if not mobile else 3.0))
    for s in series:
        v, d = last(s["col"])
        tt = (d - D0).days
        ly = pos[s["col"]]
        th = fs + (9 if not mobile else 7)
        tw = (fs + 1) * 2.6
        o.append(f'<circle cx="{X(tt):.1f}" cy="{Y(v):.1f}" r="{4 if not mobile else 3.2}" fill="{s["color"]}" stroke="var(--bg)" stroke-width="2"/>')
        o.append(f'<line x1="{X(tt)+5:.1f}" y1="{Y(v):.1f}" x2="{L+pw+8:.1f}" y2="{ly-th*0.18:.1f}" stroke="{s["color"]}" stroke-width="1" stroke-opacity="0.6"/>')
        o.append(f'<rect x="{L+pw+8}" y="{ly-th*0.18-th/2:.1f}" width="{tw:.1f}" height="{th:.1f}" rx="{th/2:.1f}" fill="{s["color"]}"/>')
        o.append(f'<text x="{L+pw+8+tw/2:.1f}" y="{ly-th*0.18+(fs+1)*0.36:.1f}" text-anchor="middle" font-size="{fs+1}" font-weight="700" fill="#FFFFFF" font-family="{NUMF}">{f1(v)}</text>')
        o.append(f'<text x="{L+pw+10}" y="{ly+th*0.5+fs*0.55:.1f}" font-size="{fs-1}" font-weight="600" fill="{s["color"]}">{esc(s["short"] if mobile else s["name"])}</text>')
    # 시작값 라벨
    starts = []
    for s in series:
        v, d = first(s["col"])
        starts.append((Y(v), s["col"]))
    pos = spread(starts, T + 10, T + ph - 4, fs + 3)
    for s in series:
        v, d = first(s["col"])
        tt = (d - D0).days
        ly = pos[s["col"]]
        o.append(f'<text x="{X(tt)+4:.1f}" y="{ly-6:.1f}" font-size="{fs-0.5}" font-weight="600" fill="{s["color"]}" font-family="{NUMF}" stroke="var(--bg)" stroke-width="3.5" paint-order="stroke">{f1(v)}</text>')
    o.append("</svg>")
    return "".join(o)


def dumbbell(rows, W, mobile=False, labs=('전화면접', '무선 ARS')):
    """rows: [(라벨, 전화면접값, ARS값)] 전화면접=채운 점, ARS=빈 점"""
    fs = 11.5 if mobile else 13
    rh = 54 if mobile else 46
    L = 12 if mobile else 130
    R = 24 if mobile else 96
    T = 34
    H = T + rh * len(rows) + 10
    vmax = max(max(a, b) for _, a, b in rows)
    vmax = math.ceil((vmax + 4) / 10) * 10
    pw = W - L - R
    X = lambda v: L + v / vmax * pw
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" style="display:block">']
    # 범례
    o.append(f'<circle cx="{L+6}" cy="12" r="6" fill="var(--navy)"/><text x="{L+16}" y="16" font-size="{fs-1}" fill="var(--ink)">{labs[0]}</text>')
    o.append(f'<circle cx="{L+16+len(labs[0])*13+20}" cy="12" r="6" fill="var(--bg)" stroke="var(--coral)" stroke-width="2"/><text x="{L+16+len(labs[0])*13+30}" y="16" font-size="{fs-1}" fill="var(--ink)">{labs[1]}</text>')
    for i, (lab, a, b) in enumerate(rows):
        cy = T + rh * i + (rh * 0.62 if mobile else rh / 2)
        if mobile:
            o.append(f'<text x="{L}" y="{cy-16}" font-size="{fs}" font-weight="600" fill="var(--ink)">{esc(lab)} <tspan fill="var(--muted)" font-weight="400">격차 {sg(rd(a, b))}%p</tspan></text>')
        else:
            o.append(f'<text x="{L-12}" y="{cy+4}" text-anchor="end" font-size="{fs}" font-weight="600" fill="var(--ink)">{esc(lab)}</text>')
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{cy}" y2="{cy}" stroke="var(--grid)"/>')
        o.append(f'<line x1="{X(min(a,b)):.1f}" x2="{X(max(a,b)):.1f}" y1="{cy}" y2="{cy}" stroke="var(--line2)" stroke-width="5" stroke-linecap="round"/>')
        o.append(f'<circle cx="{X(a):.1f}" cy="{cy}" r="7" fill="var(--navy)"/>')
        o.append(f'<circle cx="{X(b):.1f}" cy="{cy}" r="7" fill="var(--bg)" stroke="var(--coral)" stroke-width="2.4"/>')
        # 값 라벨: 큰 값은 오른쪽, 작은 값은 왼쪽
        for v, col, isa in ((a, "var(--navy)", True), (b, "var(--coral-d)", False)):
            right = v >= (b if isa else a)
            x = X(v) + (12 if right else -12)
            anc = "start" if right else "end"
            o.append(f'<text x="{x:.1f}" y="{cy+4.5}" text-anchor="{anc}" font-size="{fs}" font-weight="700" fill="{col}" font-family="{NUMF}">{f1(v)}</text>')
        if not mobile:
            o.append(f'<text x="{W-4}" y="{cy+4.5}" font-size="{fs-1}" fill="var(--muted)" text-anchor="end">격차 <tspan font-family="{NUMF}">{sg(rd(a, b))}</tspan>%p</text>')
    o.append("</svg>")
    return "".join(o)


def house_chart(items, W, mobile=False, unit_lab="%p"):
    """items: [(기관 라벨, 효과, 건수)] 내림차순 정렬 가정. 기관명과 막대를 한 줄에 맞춘다."""
    fs = 11.5 if mobile else 12.5
    rh = 30 if mobile else 30
    labs = [(lab.replace(" · 전화면접", "(전화면접)").replace(" · 무선ARS", "(ARS)"), f"{n}건") for lab, _, n in items]
    def tw(t, f):
        return sum(f * (0.95 if ord(ch) > 0x2E80 else 0.58) for ch in t)
    lw = max(tw(a, fs) + tw(" " + b, fs) for a, b in labs)
    L = math.ceil(lw) + 14
    R = 16
    T = 8
    H = T + rh * len(items) + 8
    m = max(abs(v) for _, v, _ in items)
    m = max(4, math.ceil(m + 1.5))
    pw = W - L - R
    X = lambda v: L + (v + m) / (2 * m) * pw
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" style="display:block">']
    for i in range(len(items)):
        if i % 2 == 0:
            o.append(f'<rect x="0" y="{T + rh * i}" width="{W}" height="{rh}" fill="var(--line2)" opacity=".18"/>')
    o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{T}" y2="{H-6}" stroke="var(--line2)" stroke-width="1.2"/>')
    for i, ((lab, v, n), (lt, nt)) in enumerate(zip(items, labs)):
        cy = T + rh * i + rh / 2
        o.append(f'<text x="{L-10}" y="{cy+4}" text-anchor="end" font-size="{fs}" fill="var(--ink)">{esc(lt)} <tspan fill="var(--muted)">{nt}</tspan></text>')
        col = "var(--navy)" if v >= 0 else "var(--coral)"
        x1, x2 = sorted([X(0), X(v)])
        o.append(f'<rect x="{x1:.1f}" y="{cy-6}" width="{max(1.5,x2-x1):.1f}" height="12" rx="2" fill="{col}"/>')
        tx = X(v) + (6 if v >= 0 else -6)
        o.append(f'<text x="{tx:.1f}" y="{cy+4.5}" text-anchor="{"start" if v>=0 else "end"}" font-size="{fs}" font-weight="700" fill="{col}" font-family="{NUMF}">{sg(v)}</text>')
    o.append("</svg>")
    return "".join(o)


def cv(d, m):
    return f'<div class="cv"><div class="cv-d">{d}</div><div class="cv-m">{m}</div></div>'


# ---------------------------------------------------------------- 수치 산출
app_now, d_now = last("approve")
dis_now, _ = last("disapprove")
app_0, d_0 = first("approve")
dis_0, _ = first("disapprove")
dem_now, dp = last("dem")
ppp_now, _ = last("ppp")
dem_0, _ = first("dem")
ppp_0, _ = first("ppp")
cross = None
ta, tb = MODELS["approve"]["trend"], MODELS["disapprove"]["trend"]
for i in DAYS[1:]:
    if not np.isnan(ta[i]) and not np.isnan(ta[i - 1]) and ta[i - 1] >= tb[i - 1] and ta[i] < tb[i]:
        cross = DATES[i]
cross_p = None
pa, pb = MODELS["dem"]["trend"], MODELS["ppp"]["trend"]
gap_max_i = int(np.nanargmax(pa - pb))

n_polls = len(df)
n_orgs = df["org"].nunique()

# 방법별(최근 30일, 기관별 평균 후 평균)
recent = df[df["mid"] >= LAST_FIELD - pd.Timedelta(days=30)]
def by_method(col):
    out = {}
    for mth in ["전화면접", "무선ARS"]:
        s = recent[(recent["method"] == mth) & recent[col].notna()]
        if len(s):
            out[mth] = (s.groupby("org")[col].mean().mean(), s["org"].nunique(), len(s))
    return out
MCOLS = [("approve", "국정 긍정평가"), ("disapprove", "국정 부정평가"), ("dem", "더불어민주당"),
         ("ppp", "국민의힘"), ("none", "지지 정당 없음")]
mrows = []
for c, lab in MCOLS:
    bm = by_method(c)
    if "전화면접" in bm and "무선ARS" in bm:
        mrows.append((lab, bm["전화면접"][0], bm["무선ARS"][0]))
mrows.sort(key=lambda r: -abs(r[1] - r[2]))
# 의뢰 주체별(ARS 안에서 비교, 방법 차이를 빼기 위함)
ars = recent[recent["method"] == "무선ARS"]
srows = []
for c, lab in MCOLS:
    a1 = ars[(ars["sp_type"] == "자체조사") & ars[c].notna()]
    b1 = ars[(ars["sp_type"] == "언론사 의뢰") & ars[c].notna()]
    if len(a1) and len(b1):
        srows.append((lab, a1.groupby("org")[c].mean().mean(), b1.groupby("org")[c].mean().mean()))
srows.sort(key=lambda r: -abs(r[1] - r[2]))
sp_self = sorted(ars[ars["sp_type"] == "자체조사"]["org"].unique())
sp_media = sorted(ars[ars["sp_type"] == "언론사 의뢰"]["org"].unique())
phone_orgs = recent[recent["method"] == "전화면접"]["org"].nunique()
ars_orgs = recent[recent["method"] == "무선ARS"]["org"].nunique()

# 기관별 표
H_APP = MODELS["approve"]["house"]
H_DEM = MODELS["dem"]["house"]
H_PPP = MODELS["ppp"]["house"]
orgs = []
for u, g in df.groupby("unit"):
    orgs.append(dict(unit=u, org=g["org"].iloc[0], method=g["method"].iloc[0],
                     sp="·".join(sorted(set(x for x in g["sp_type"] if x != "미기재"))) or "미기재",
                     n_app=int(g["approve"].notna().sum()), n_party=int(g["dem"].notna().sum()),
                     n=float(g["sample_n"].mean()), rr=float(g["response_rate"].mean()) if g["response_rate"].notna().any() else float("nan"),
                     h_app=H_APP.get(u, float("nan")), h_dem=H_DEM.get(u, float("nan")), h_ppp=H_PPP.get(u, float("nan")),
                     none=float(g["none"].mean()) if g["none"].notna().any() else float("nan")))
orgs.sort(key=lambda r: -(r["h_app"] if not math.isnan(r["h_app"]) else -99))
hitems = [(o["unit"], o["h_app"], o["n_app"]) for o in orgs if not math.isnan(o["h_app"])]
hitems_p = sorted([(o["unit"], o["h_dem"] - o["h_ppp"], o["n_party"]) for o in orgs if not math.isnan(o["h_dem"]) and not math.isnan(o["h_ppp"])], key=lambda r: -r[1])

# 문장
def per(d):
    return f"{d.month}월 " + ("초" if d.day <= 10 else "중순" if d.day <= 20 else "하순")

gap_now = app_now - dis_now
if gap_now < 0:
    HEADLINE = f"국정수행 부정평가 {f1(dis_now)}%, 긍정평가보다 {f1(rd(dis_now, app_now))}%p 높아"
else:
    HEADLINE = f"국정수행 긍정평가 {f1(app_now)}%, 부정평가보다 {f1(rd(app_now, dis_now))}%p 높아"
DECK = f"정당지지도는 더불어민주당 {f1(dem_now)}%, 국민의힘 {f1(ppp_now)}%로 차이 {f1(abs(rd(dem_now, ppp_now)))}%p"

# ---------------------------------------------------------------- 표
def th(cols):
    return "".join(f"<th>{c}</th>" for c in cols)

org_rows = []
for o in orgs:
    org_rows.append(
        f"<tr><td class='lab'>{esc(o['org'])}</td><td>{esc(o['method'])}</td><td>{o['sp']}</td>"
        f"<td class='num'>{o['n_app']}</td><td class='num'>{o['n_party']}</td><td class='num'>{f0(o['n'])}</td>"
        f"<td class='num'>{f1(o['rr'])}</td>"
        f"<td class='num {'pos' if o['h_app']>0 else 'neg'}'>{sg(o['h_app']) if not math.isnan(o['h_app']) else ''}</td>"
        f"<td class='num'>{sg(o['h_dem']) if not math.isnan(o['h_dem']) else ''}</td>"
        f"<td class='num'>{sg(o['h_ppp']) if not math.isnan(o['h_ppp']) else ''}</td>"
        f"<td class='num'>{f1(o['none'])}</td></tr>")

poll_rows = []
for _, r in df.sort_values(["field_end", "org"], ascending=[False, True]).iterrows():
    link = f"<a href='{esc(r['source_url'])}' target='_blank' rel='noopener'>보도</a>" if isinstance(r["source_url"], str) else ""
    poll_rows.append(
        f"<tr><td class='num'>{sdate(r['field_start'])}~{sdate(r['field_end'])}</td><td class='lab'>{esc(r['org'])}</td>"
        f"<td class='hm lab'>{esc(r['sp_short'])}</td><td>{'전화' if r['method']=='전화면접' else 'ARS'}</td>"
        f"<td class='num'>{f0(r['sample_n'])}</td><td class='num hm'>{f1(r['response_rate'])}</td>"
        f"<td class='num'>{f1(r['approve'])}</td><td class='num'>{f1(r['disapprove'])}</td>"
        f"<td class='num'>{f1(r['dem'])}</td><td class='num'>{f1(r['ppp'])}</td><td class='num hm'>{f1(r['none'])}</td>"
        f"<td class='hm'>{link}</td></tr>")

# ---------------------------------------------------------------- 차트 생성
EV = [("2026-09-24", "2026-09-26", "추석")]
# 추세가 꺾인 구간의 주요 사건(보도로 날짜 확인)
MARKS = [("2026-08-13", "8·13 주택공급대책 발표, 용산공원 부지 논란"),
         ("2026-08-17", "더불어민주당 전당대회"),
         ("2026-08-30", "6개 부처 개각, 인선 논란"),
         ("2026-09-09", "호르무즈 해협 파병 검토 논란"),
         ("2026-09-18", "대통령 기자회견")]

def ev_list(marks):
    li = "".join(f'<li><span class="evn">{k}</span><span class="evd">{pd.Timestamp(d).month}.{pd.Timestamp(d).day}</span>{esc(t)}</li>'
                 for k, (d, t) in enumerate(marks, 1))
    return f'<ol class="evl">{li}</ol>'
EV_HTML = ev_list(MARKS)
S_APP = [dict(col="approve", name="국정 긍정", short="긍정", color=C_APP),
         dict(col="disapprove", name="국정 부정", short="부정", color=C_DIS)]
S_BIG = [dict(col="dem", name="더불어민주당", short="민주", color=PARTIES[0][2]),
         dict(col="ppp", name="국민의힘", short="국힘", color=PARTIES[1][2])]
S_SMALL = [dict(col=c, name=n, short={"rebuild": "조국혁신", "reform": "개혁신당", "progressive": "진보당"}[c], color=k)
           for c, n, k in PARTIES[2:]]

def rng(cols, step):
    vals = pd.concat([df[c].dropna() for c in cols])
    lo = math.floor((vals.min() - 1) / step) * step
    hi = math.ceil((vals.max() + 1) / step) * step
    return max(0, lo), hi

a_lo, a_hi = rng(["approve", "disapprove"], 10)
p_lo, p_hi = rng(["dem", "ppp"], 5)
CH_APP = cv(line_chart(S_APP, a_lo, a_hi, 10, 860, 400, events=EV, marks=MARKS, cross_at=cross),
            line_chart(S_APP, a_lo, a_hi, 10, 350, 300, mobile=True, events=EV, marks=MARKS, cross_at=cross))
CH_BIG = cv(line_chart(S_BIG, p_lo, p_hi, 5, 410, 340, events=EV, marks=MARKS),
            line_chart(S_BIG, p_lo, p_hi, 5, 350, 290, mobile=True, events=EV, marks=MARKS))
CH_SMALL = cv(line_chart(S_SMALL, 0, 8, 2, 410, 340, events=EV, marks=MARKS),
              line_chart(S_SMALL, 0, 8, 2, 350, 290, mobile=True, events=EV, marks=MARKS))
CH_METH = cv(dumbbell(mrows, 570), dumbbell(mrows, 350, mobile=True))
CH_SP = cv(dumbbell(srows, 570, labs=('자체조사', '언론사 의뢰')), dumbbell(srows, 350, mobile=True, labs=('자체조사', '언론사 의뢰')))
CH_HOUSE = cv(house_chart(hitems, 410), house_chart(hitems, 350, mobile=True))
CH_HOUSE_P = cv(house_chart(hitems_p, 570), house_chart(hitems_p, 350, mobile=True))

# 본문 문장(기사체)
def jo(w, a, b):
    c = ord(w[-1]) - 0xAC00
    return a if 0 <= c <= 11171 and c % 28 else b
def mv(col):
    v1, _ = last(col); v0, _ = first(col)
    return rd(v1, v0)
max_gap_d = DATES[gap_max_i]
none_m = by_method("none")
def un(u):
    o, m = u.split(' · ')
    return f"{o}({'ARS' if 'ARS' in m else m})"
top_h = hitems[0]; bot_h = hitems[-1]
tp = hitems_p[0]; bp = hitems_p[-1]
ppp_tr = MODELS["ppp"]["trend"]; ppp_min_i = int(np.nanargmin(ppp_tr))
small = []
for c, n, _ in PARTIES[2:]:
    small.append((last(c)[0], mv(c), n, c))
small_sorted = sorted(small, reverse=True)
riser = max(small, key=lambda r: r[1])
n_ph = phone_orgs; n_ars = ars_orgs
d_s = df["field_start"].min()

LEAD = (f"{LAST_FIELD.month}월 {LAST_FIELD.day}일까지 실시돼 중앙선거여론조사심의위원회에 등록된 전국 단위 여론조사 "
        f"{n_polls:,}건을 종합한 결과, 대통령 국정수행에 대한 부정평가는 {f1(dis_now)}%, 긍정평가는 {f1(app_now)}%로 집계됐다. "
        f"{per(d_0)} 긍정 {f1(app_0)}%, 부정 {f1(dis_0)}%였던 평가는 "
        + (f"{per(cross)} 뒤집힌 뒤 격차가 벌어졌다. " if cross else "같은 흐름을 유지했다. ")
        + f"정당지지도는 더불어민주당 {f1(dem_now)}%, 국민의힘 {f1(ppp_now)}%로 나타났다.")

APP_TITLE = (f"부정평가, {per(cross)} 긍정평가 앞선 뒤 격차 확대" if cross and gap_now < 0
             else "국정수행 긍정·부정평가 추이")
T_APP = (f"긍정평가는 {per(d_0)} {f1(app_0)}%에서 {per(d_now)} {f1(app_now)}%로 {f1(abs(mv('approve')))}%p "
         f"{'내렸고' if mv('approve') < 0 else '올랐고'}, 부정평가는 {f1(dis_0)}%에서 {f1(dis_now)}%로 "
         f"{f1(abs(mv('disapprove')))}%p {'올랐다' if mv('disapprove') > 0 else '내렸다'}. "
         + (f"두 추세선은 {kdate(cross)} 교차했다. " if cross else "")
         + "같은 시기 개별 조사 결과는 기관과 조사방법에 따라 위아래로 흩어져 있다.")

PARTY_TITLE = f"민주당 {f1(dem_now)}%, 국민의힘 {f1(ppp_now)}%"
T_PARTY = (f"민주당 우위는 {kdate(max_gap_d)} {f1(float(pa[gap_max_i]-pb[gap_max_i]))}%p로 가장 컸다. "
           f"국민의힘은 {kdate(DATES[ppp_min_i])} {f1(float(ppp_tr[ppp_min_i]))}%까지 내려간 뒤 9월 들어 반등해 "
           f"{per(dp)} {f1(ppp_now)}%를 기록했고, 양당 차이는 {f1(abs(rd(dem_now, ppp_now)))}%p로 좁혀졌다. "
           "양당 지지율 수준은 전화면접 조사와 ARS 조사의 중간에 오도록 맞춘 값이어서, 수준보다 변화 방향을 보는 데 적합하다.")

SMALL_TITLE = ", ".join(f"{n} {f1(v)}%" for v, _, n, _ in small_sorted)
T_SMALL = (f"{riser[2]}는 {per(d_0)} {f1(first(riser[3])[0])}%에서 {f1(riser[0])}%로 {f1(riser[1])}%p 올라 군소 정당 가운데 상승 폭이 가장 컸다. "
           "지지율 1~2%대 정당은 조사 간 차이가 표본오차 범위 안에 있어 작은 변화에는 의미를 두기 어렵다.")
T_SMALL = T_SMALL.replace(f"{riser[2]}는", f"{riser[2]}{jo(riser[2],'은','는')}")

# 조사방법별 표(항목 순서 고정)
MBM = {c: by_method(c) for c, _ in MCOLS}
METH_ROWS = ""
for c, lab in MCOLS:
    bm = MBM[c]
    if "전화면접" in bm and "무선ARS" in bm:
        p_, a_ = bm["전화면접"][0], bm["무선ARS"][0]
        dv = rd(a_, p_)
        cls = "big" if abs(dv) >= 5 else ""
        METH_ROWS += (f"<tr><td class='lab'>{lab}</td><td class='cn'><span class='nb'>{f1(p_)}</span></td><td class='cn'><span class='nb'>{f1(a_)}</span></td>"
                      f"<td class='cn {cls}'><span class='nb'>{sg(dv)}</span></td></tr>")
nb = MBM["none"]
METH_TITLE = f"ARS 조사, 무당층 적고 국민의힘 지지율 높게 나와"
T_METH = (f"지지 정당이 없다는 응답은 전화면접 조사가 {f1(nb['전화면접'][0])}%로 ARS 조사({f1(nb['무선ARS'][0])}%)보다 "
          f"{f1(abs(rd(nb['전화면접'][0], nb['무선ARS'][0])))}%p 높았다. 차이가 5%p 이상인 항목은 굵게 표시했다.")
HOUSE_TITLE = f"같은 기간에도 기관에 따라 긍정평가 최대 {f1(rd(top_h[1], bot_h[1]))}%p 차이"
T_HOUSE = (f"종합 추세와 비교해 긍정평가가 가장 높게 나온 곳은 {un(top_h[0])}{jo(un(top_h[0])[:-1],'으로','로')} {f1(abs(top_h[1]))}%p 높았고, "
           f"가장 낮게 나온 곳은 {un(bot_h[0])}{jo(un(bot_h[0])[:-1],'으로','로')} {f1(abs(bot_h[1]))}%p 낮았다. "
           "조사 건수가 적은 기관은 차이를 작게 잡아 추정했다.")

HOUSE_P_TITLE = "양당 격차도 기관에 따라 다르게 나타나"
T_HOUSE_P = (f"민주당과 국민의힘의 격차는 {un(tp[0])}에서 종합 추세보다 {f1(abs(tp[1]))}%p 크게, "
             f"{un(bp[0])}에서 {f1(abs(bp[1]))}%p 작게 나왔다. 같은 기관이라도 전화면접과 ARS 조사는 따로 계산했다.")

big = srows[0] if srows else None
SP_TITLE = (f"ARS 조사, 의뢰 주체에 따라 {big[0]} {f1(abs(rd(big[1], big[2])))}%p 차이" if big else "의뢰 주체별 비교")
T_SP = (f"ARS 조사만 놓고 자체조사({', '.join(sp_self)})와 언론사 의뢰 조사({', '.join(sp_media)})를 비교했다. "
        + (f"차이가 가장 큰 항목은 {big[0]}로 자체조사 {f1(big[1])}%, 언론사 의뢰 조사 {f1(big[2])}%였다. " if big else "")
        + "비교 대상 기관이 적어 의뢰 주체의 영향과 기관별 차이를 구분하기는 어렵다.")
T_SP = T_SP.replace(f"{big[0]}로", f"{big[0]}{jo(big[0],'으로','로')}") if big else T_SP

def blk(text):
    return f'<p class="note">{esc(text)}</p>'

# ---------------------------------------------------------------- 폰트
try:
    font_b64 = base64.b64encode(open(FONT, "rb").read()).decode()
except FileNotFoundError:
    font_b64 = ""  # 숫자 글꼴 파일이 없으면 시스템 글꼴로 표시

# ---------------------------------------------------------------- 주별 격차 띠(머리말 아래)
def gap_strip():
    ta_, tb_ = MODELS["approve"]["trend"], MODELS["disapprove"]["trend"]
    valid = [i for i in DAYS if not np.isnan(ta_[i]) and not np.isnan(tb_[i])]
    i0, i1 = valid[0], valid[-1]
    pts = list(range(i1, i0 - 1, -7))[::-1]
    cells = []
    for i in pts:
        g = rd(tb_[i], ta_[i])          # 부정 - 긍정
        a_ = min(abs(g) / 20, 1)
        if g > 0:
            bg = f"rgba(198,69,43,{0.12 + 0.78 * a_:.2f})"
        else:
            bg = f"rgba(31,122,92,{0.12 + 0.78 * a_:.2f})"
        fg = "#FFFFFF" if a_ > 0.45 else "var(--ink)"
        d = DATES[i]
        cells.append(f'<div class="gc" style="background:{bg};color:{fg}" title="{d.month}월 {d.day}일 기준 부정 {f1(tb_[i])}%, 긍정 {f1(ta_[i])}%">'
                     f'<b>{sg(g)}</b><span>{d.month}.{d.day}</span></div>')
    return ('<div class="gap"><div class="gaph"><span class="gt">주별 국정평가 격차</span>'
            '<span class="gl"><i class="gp"></i>긍정 우위 <i class="gn"></i>부정 우위 · 부정에서 긍정을 뺀 값(%p), 종합 추세 기준</span></div>'
            f'<div class="gs">{"".join(cells)}</div></div>')
GAP_HTML = gap_strip()

# ---------------------------------------------------------------- 로고(있으면 내장, 없으면 글자 표기)
import os
# 서던포스트 국문 기본판(southernpost-ci 스킬 assets/ko.png) 내장 - 파일이 없어도 로고가 들어가게
LOGO_B64 = "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"
LOGO = next((p for p in ["logo.png"] if os.path.exists(p)), None)
_lb = base64.b64encode(open(LOGO, "rb").read()).decode() if LOGO else LOGO_B64
if _lb:
    BRAND = f'<img class="logo" src="data:image/png;base64,{_lb}" alt="서던포스트">'
    BRAND_FOOT = f'<img class="logo" src="data:image/png;base64,{_lb}" alt="서던포스트">'
else:
    BRAND = '<span class="wordmark">서던포스트</span>'
    BRAND_FOOT = '<span class="wordmark">서던포스트</span>'

# ---------------------------------------------------------------- HTML
build_time = dt.datetime.now().strftime("%Y.%m.%d")

page = f"""<title>여론조사 종합 추세</title>
<style>
/* 레이아웃: 기사형 머리말 + 주별 격차 띠 → 왼쪽 섹션 레일(번호·제목 고정) + 오른쪽 본문, 상자 대신 괘선으로 구획, 흰 바탕 고정 */
__FONTFACE__
:root{{
  color-scheme:light;
  --bg:#FFFFFF; --soft:#F5F7FA; --ink:#121A26; --body:#2B3544; --muted:#6C7787; --line:#E2E6EC; --line2:#C6CED9; --grid:#EEF1F5;
  --navy:#1F3354; --navy-s:#E8EDF5; --coral:#D9502F; --coral-d:#C6452B;
  --gold:#A8731F; --gold-s:#FBF3E4; --app:{C_APP}; --dis:{C_DIS}; --dem:{PARTIES[0][2]}; --ppp:{PARTIES[1][2]};
  --ev:#4F5B6B; --sans:{FONT_STACK}; --num:{NUMF};
}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;word-break:keep-all;overflow-wrap:anywhere;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}}
svg text{{font-variant-numeric:tabular-nums}}
a{{color:var(--navy)}}
:focus-visible{{outline:2px solid var(--coral);outline-offset:2px}}
header.top{{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line)}}
.bar{{max-width:1160px;margin:0 auto;padding:13px 28px;display:flex;align-items:center;gap:22px;flex-wrap:wrap}}
.logo{{display:block;width:104px;height:auto;object-fit:contain}}
.wordmark{{font-weight:800;font-size:17px;color:var(--navy);letter-spacing:-.01em}}
.nav{{display:flex;gap:2px;flex-wrap:wrap;margin-left:auto}}
.nav a{{font-size:13.5px;font-weight:600;color:var(--muted);text-decoration:none;padding:5px 10px;border-radius:6px}}
.nav a:hover{{color:var(--ink);background:var(--soft)}}
.src{{font-size:12.5px;color:var(--muted);padding-left:14px;border-left:1px solid var(--line)}}
.wrap{{max-width:1160px;margin:0 auto;padding-inline:28px;padding-block:48px 72px}}
.mast{{padding-bottom:8px}}
.kicker{{display:inline-block;font-size:12.5px;font-weight:700;color:var(--navy);border:1px solid var(--navy);border-radius:4px;padding:2px 9px;margin-bottom:16px;letter-spacing:.02em}}
.headline{{margin:0 0 12px;font-size:42px;font-weight:800;line-height:1.24;letter-spacing:-.03em;text-wrap:balance;max-width:22em}}
.deck{{font-size:21px;font-weight:600;color:var(--navy);margin:0 0 22px;line-height:1.45;text-wrap:balance;letter-spacing:-.01em}}
.leadp{{font-size:17px;line-height:1.85;max-width:44em;margin:0;color:var(--body)}}
.meta{{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:20px;font-size:13px;color:var(--muted)}}
.meta b{{color:var(--ink);font-weight:700}}
.gap{{margin-top:34px;padding:18px 0 0;border-top:1px solid var(--line)}}
.gaph{{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 16px;margin-bottom:10px}}
.gt{{font-size:14px;font-weight:700}}
.gl{{font-size:12.5px;color:var(--muted)}}
.gl i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 8px;vertical-align:-1px}}
.gl .gp{{background:rgba(31,122,92,.75);margin-left:0}} .gl .gn{{background:rgba(198,69,43,.75)}}
.gs{{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(46px,1fr);gap:3px;overflow-x:auto;padding-bottom:2px}}
.gc{{padding:9px 2px 7px;text-align:center;border-radius:4px;line-height:1.25}}
.gc b{{display:block;font-size:14px;font-weight:700}}
.gc span{{display:block;font-size:11px;opacity:.85;margin-top:2px}}
.part{{display:grid;grid-template-columns:200px minmax(0,1fr);gap:0 40px;margin-top:64px;padding-top:22px;border-top:2px solid var(--ink);scroll-margin-top:70px}}
.rail{{align-self:start;position:sticky;top:78px}}
.rail .sn{{display:block;font-size:40px;font-weight:300;line-height:1;color:var(--coral);letter-spacing:-.02em;margin-bottom:10px}}
.rail h2{{margin:0 0 6px;font-size:21px;font-weight:800;letter-spacing:-.02em;line-height:1.3}}
.rail .sd{{margin:0;font-size:13.5px;color:var(--muted);line-height:1.6}}
.pbody{{min-width:0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:0}}
.card{{min-width:0;padding:0 0 30px}}
.card.full{{grid-column:1/-1}}
.card.half{{padding-top:26px;border-top:1px solid var(--line)}}
.card.half.l{{padding-right:28px}}
.card.half.r{{padding-left:28px;border-left:1px solid var(--line)}}
.grid > .card.half:first-child,.grid > .card.half:first-child + .card.half{{border-top:0;padding-top:0}}
.pbody > .card + .card{{padding-top:26px;border-top:1px solid var(--line)}}
.eyebrow{{font-size:12.5px;font-weight:700;letter-spacing:.02em;color:var(--coral-d)}}
.ctitle{{font-size:21px;font-weight:800;line-height:1.38;margin:4px 0 6px;letter-spacing:-.02em;text-wrap:balance}}
.half .ctitle{{font-size:18.5px}}
.csub{{font-size:13px;color:var(--muted);margin-bottom:14px}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--muted);margin:-4px 0 6px}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}}
.legend .sp,.legend .sa{{width:18px;border-radius:2px;background:var(--muted);vertical-align:3px}} .legend .sp{{height:3.5px;opacity:.7}} .legend .sa{{height:1.8px;opacity:.6}}
.note{{margin:16px 0 0;font-size:15px;line-height:1.8;color:var(--body);max-width:52em}}
.foot{{font-size:12px;color:var(--muted);margin-top:12px}}
.cv-m{{display:none}}
.tw{{overflow-x:auto;margin-top:4px}}
table.t{{width:100%;border-collapse:collapse;table-layout:fixed;font-size:13.5px}}
table.t th{{font-weight:700;text-align:center;padding:9px 6px;border-bottom:1.5px solid var(--ink);word-break:keep-all;vertical-align:bottom;color:var(--ink)}}
table.t td{{padding:8px 6px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}}
table.t td.num{{font-variant-numeric:tabular-nums;text-align:right;padding-right:14px}}
table.t td.lab{{text-align:left;white-space:normal;word-break:keep-all}}
table.t tbody tr:hover td{{background:var(--soft)}}
table.t tr:last-child td{{border-bottom:1.5px solid var(--ink)}}
table.t.mt td.lab{{font-weight:600}} table.t td.cn{{text-align:center}} .nb{{display:inline-block;min-width:3.4em;text-align:right;font-variant-numeric:tabular-nums}} table.t td.big{{font-weight:800;color:var(--coral-d)}}
.evl{{list-style:none;margin:14px 0 0;padding:12px 0 0;border-top:1px dashed var(--line2);display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;color:var(--body)}}
.evl li{{display:flex;align-items:baseline;gap:7px;min-width:0}}
.evn{{flex:none;display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;background:var(--ev);color:#FFFFFF;font-size:11px;font-weight:700;transform:translateY(-1px)}}
.evd{{flex:none;font-weight:700;color:var(--ev)}}
.method dl{{display:grid;grid-template-columns:120px 1fr;gap:12px 20px;margin:0;font-size:14.5px;line-height:1.7}}
.method dt{{font-weight:700;color:var(--navy)}} .method dd{{margin:0;color:var(--body)}}
footer{{border-top:2px solid var(--ink);margin-top:24px}}
footer .bar{{padding:22px 28px;font-size:12.5px;color:var(--muted);gap:16px}}
footer .logo{{width:92px;height:auto}}
@media(max-width:1000px){{.part{{grid-template-columns:1fr}} .rail{{position:static;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;margin-bottom:18px}} .rail .sn{{font-size:26px;margin:0}} .rail .sd{{flex-basis:100%}}}}
@media(max-width:860px){{.grid{{grid-template-columns:1fr}} .card.half.l,.card.half.r{{padding-left:0;padding-right:0;border-left:0}} .nav{{display:none}} .src{{margin-left:auto}}}}
@media(max-width:640px){{
  .bar{{padding:10px 16px;gap:10px}} .wrap{{padding-inline:16px;padding-block:28px 44px}}
  .logo{{width:88px;height:auto}} .src{{font-size:11.5px}}
  .headline{{font-size:26px}} .deck{{font-size:17px}} .leadp{{font-size:15.5px;line-height:1.8}}
  .part{{margin-top:44px}} .rail h2{{font-size:19px}}
  .ctitle,.half .ctitle{{font-size:17px}} .note{{font-size:14.5px}}
  .gs{{grid-auto-flow:row;grid-template-columns:repeat(8,1fr);overflow:visible}} .gc{{padding:7px 0 6px}} .gc b{{font-size:12px;letter-spacing:-.02em}} .gc span{{font-size:10px}}
  .cv-d{{display:none}} .cv-m{{display:block}}
  table.t{{font-size:12px}} table.t .hm{{display:none}}
  table.t td.num{{padding-right:6px}}
  .method dl{{grid-template-columns:1fr;gap:4px 0}} .method dd{{margin-bottom:8px}}
}}
@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
</style>

<header class="top"><div class="bar">{BRAND}<nav class="nav" aria-label="섹션 이동"><a href="#trend">종합 추세</a><a href="#type">유형별 비교</a><a href="#source">자료 출처</a><a href="#method">산출 방법</a></nav><div class="src">{build_time} 갱신</div></div></header>

<main class="wrap">
  <div class="mast">
  <div class="kicker">여론조사 종합 추세</div>
  <h1 class="headline">{esc(HEADLINE)}</h1>
  <p class="deck">{esc(DECK)}</p>
  <p class="leadp">{esc(LEAD)}</p>
  <div class="meta"><span>분석 조사 <b>{n_polls:,}건</b></span><span>조사기관 <b>{n_orgs}곳</b></span><span>조사기간 <b>{d_s.month}.{d_s.day}~{LAST_FIELD.month}.{LAST_FIELD.day}</b></span></div>
  </div>

  <section class="part" id="trend"><aside class="rail"><span class="sn">01</span><h2>종합 추세</h2><p class="sd">기관별 차이를 보정한 가중 평균</p></aside><div class="pbody">
  <div class="grid">
    <section class="card full">
      <div class="eyebrow">대통령 국정수행 평가</div>
      <h2 class="ctitle">{esc(APP_TITLE)}</h2>
      <div class="csub">단위: % · 굵은 선은 종합 추세, 짧은 가로선은 개별 조사(선 길이가 조사기간), 두 선 사이 색은 앞선 쪽, 번호는 주요 사건 시점</div>
      <div class="legend"><span><i class="sp"></i>전화면접 조사</span><span><i class="sa"></i>무선 ARS 조사</span></div>
      {CH_APP}
      {EV_HTML}
      {blk(T_APP)}
      <div class="foot">※ 조사의뢰자·조사기관·조사기간은 3장 목록 참조. 자세한 내용은 중앙선거여론조사심의위원회 홈페이지 참조</div>
    </section>
    <section class="card half l">
      <div class="eyebrow">정당지지도 · 양대 정당</div>
      <h2 class="ctitle">{esc(PARTY_TITLE)}</h2>
      <div class="csub">단위: % · 기관별 차이를 보정한 종합 추세 · 번호는 위 차트의 주요 사건</div>
      {CH_BIG}
      {blk(T_PARTY)}
      <div class="foot">※ 조사의뢰자·조사기관·조사기간은 3장 목록 참조. 자세한 내용은 중앙선거여론조사심의위원회 홈페이지 참조</div>
    </section>
    <section class="card half r">
      <div class="eyebrow">정당지지도 · 군소 정당</div>
      <h2 class="ctitle">{esc(SMALL_TITLE)}</h2>
      <div class="csub">단위: % · 세로축 0~8% 구간 확대 · 번호는 위 차트의 주요 사건</div>
      {CH_SMALL}
      {blk(T_SMALL)}
    </section>
  </div>

  </div></section>

  <section class="part" id="type"><aside class="rail"><span class="sn">02</span><h2>유형별 비교</h2><p class="sd">조사방법과 조사기관에 따른 차이</p></aside><div class="pbody">
  <div class="grid">
    <section class="card half l">
      <div class="eyebrow">조사방법별</div>
      <h2 class="ctitle">{esc(METH_TITLE)}</h2>
      <div class="csub">단위: %, %p · 최근 30일 조사 · 전화면접 {n_ph}개, ARS {n_ars}개 기관 평균</div>
      <div class="tw"><table class="t mt">
        <colgroup><col style="width:34%"><col style="width:22%"><col style="width:22%"><col style="width:22%"></colgroup>
        <thead><tr><th>항목</th><th>전화면접</th><th>무선 ARS</th><th>차이<br>(ARS-전화)</th></tr></thead>
        <tbody>{METH_ROWS}</tbody></table></div>
      {blk(T_METH)}
    </section>
    <section class="card half r">
      <div class="eyebrow">조사기관별 · 국정 긍정평가</div>
      <h2 class="ctitle">{esc(HOUSE_TITLE)}</h2>
      <div class="csub">단위: %p · 종합 추세보다 높게(+) 또는 낮게(-) 나온 정도</div>
      {CH_HOUSE}
      {blk(T_HOUSE)}
    </section>
  </div>

  </div></section>

  <section class="part" id="source"><aside class="rail"><span class="sn">03</span><h2>자료 출처와 개별 조사</h2><p class="sd">무엇을 어떻게 참조했는지</p></aside><div class="pbody">
  <section class="card method">
    <div class="eyebrow">자료 출처</div>
    <h2 class="ctitle">자료 출처와 인용 기준</h2>
    <div class="csub">이 페이지의 수치는 서던포스트가 직접 조사한 결과가 아니다</div>
    <dl>
      <dt>대상 조사</dt><dd>중앙선거여론조사심의위원회에 등록된 전국 단위 조사 가운데 {d_s.month}월 {d_s.day}일부터 {LAST_FIELD.month}월 {LAST_FIELD.day}일까지 실시된 조사의 대통령 국정수행 평가와 정당지지도 문항이다.</dd>
      <dt>참조 경로</dt><dd>등록 조사 결과를 인용한 언론 보도와 조사기관이 공개한 결과 보고서에서 수치를 옮겼다.</dd>
      <dt>옮긴 항목</dt><dd>조사의뢰자, 조사기관, 조사기간, 조사방법, 표본 크기, 표본오차, 응답률, 문항별 결과다.</dd>
      <dt>옮긴 방식</dt><dd>한 조사를 여러 매체가 보도한 경우 항목이 가장 많이 실린 자료를 기준으로 하고, 빠진 항목만 다른 자료로 보충했다. 어느 자료에도 없는 항목은 빈칸으로 두었다.</dd>
      <dt>확인 범위</dt><dd>심의위원회 결과분석 원자료와는 대조하지 않았다. 공식 인용 때는 원자료를 확인해야 한다.</dd>
      <dt>인용 표기</dt><dd>개별 조사의 조사의뢰자·조사기관·조사기간은 아래 목록에 적었다. 자세한 내용은 중앙선거여론조사심의위원회 홈페이지(www.nesdc.go.kr)를 참조하면 된다.</dd>
    </dl>
  </section>
  <section class="card">
    <h2 class="ctitle">분석 대상 조사 {n_polls:,}건</h2>
    <div class="csub">단위: %, 명 · 최근 조사 순 · 빈칸은 해당 문항이 없거나 보도되지 않은 경우</div>
    <div class="tw"><table class="t">
      <colgroup><col style="width:11%"><col style="width:15%"><col class="hm" style="width:12%"><col style="width:6%"><col style="width:8%"><col class="hm" style="width:7%"><col span="4" style="width:7%"><col class="hm" style="width:7%"><col class="hm" style="width:6%"></colgroup>
      <thead><tr><th>조사기간</th><th>조사기관</th><th class="hm">조사의뢰자</th><th>방법</th><th>표본</th><th class="hm">응답률</th><th>긍정</th><th>부정</th><th>민주</th><th>국힘</th><th class="hm">무당층</th><th class="hm">출처</th></tr></thead>
      <tbody>{''.join(poll_rows)}</tbody></table></div>
    <div class="foot">※ 자세한 내용은 중앙선거여론조사심의위원회 홈페이지 참조. 출처는 수치를 옮긴 보도 또는 조사기관 공개 자료</div>
  </section>

  </div></section>

  <section class="part" id="method"><aside class="rail"><span class="sn">04</span><h2>산출 방법</h2><p class="sd">보정·가중·추세 계산 기준</p></aside><div class="pbody">
  <section class="card method">
    <dl>
      <dt>기관별 차이 보정</dt><dd>기관과 조사방법 조합별로 종합 추세와의 평균 차이를 추정해 뺐다. 조사 건수가 적은 기관은 차이를 작게 잡았고(가상 조사 2건 반영), 전체 기관의 차이 평균은 0이 되도록 맞췄다.</dd>
      <dt>가중치</dt><dd>표본 크기(√(n/1,000), n은 최대 3,000), 조사방법(전화면접 1.0, ARS 0.9), 같은 기관의 조사 빈도(10일 안에 실시된 같은 기관 조사 수의 역제곱근)를 곱해 썼다.</dd>
      <dt>종합 추세</dt><dd>조사기간 중간일을 기준으로 앞뒤 21일 안의 조사를 7일 폭 정규분포 가중치로 평균했다. 마지막 조사 이후 시점은 추정하지 않았다.</dd>
      <dt>유형별 비교</dt><dd>기관별 평균을 먼저 구한 뒤 다시 평균해 매주 조사하는 기관의 비중이 커지지 않게 했다.</dd>
      <dt>유의 사항</dt><dd>리얼미터는 국정수행 평가와 정당지지도를 표본과 기간이 다른 별도 조사로 실시한다. 무당층에 '모름'을 포함하는지는 기관마다 다르다. 표본과 응답률 중 보도되지 않은 값은 빈칸으로 두었다.</dd>
    </dl>
  </section>
  </div></section>
</main>
<footer><div class="bar">{BRAND_FOOT}<span>개별 조사의 자세한 내용은 중앙선거여론조사심의위원회 홈페이지 참조 · 수치는 원자료 소수점으로 계산한 뒤 표시 단계에서 반올림</span></div></footer>
"""
page = page.replace("—", "-").replace("–", "-")
# Pretendard: 페이지에 쓰인 글자만 잘라 내장(외부 요청 없음)
import subprocess, tempfile
chars = set(page) | set(chr(c) for c in range(32, 127)) | set("·~±%▲▼")
tf = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8"); tf.write("".join(sorted(chars))); tf.close()
faces = []
for w, fn in [(400, "P-Regular.otf"), (600, "P-SemiBold.otf"), (700, "P-Bold.otf"), (800, "P-ExtraBold.otf")]:
    if not os.path.exists(fn):
        continue
    out = f"_sub_{w}.woff2"
    subprocess.run(["pyftsubset", fn, f"--text-file={tf.name}", "--flavor=woff2", f"--output-file={out}",
                    "--layout-features=kern,tnum,liga"], check=True)
    b64 = base64.b64encode(open(out, "rb").read()).decode()
    faces.append(f"@font-face{{font-family:'PretendardSub';src:url(data:font/woff2;base64,{b64}) format('woff2');font-weight:{w};font-display:swap}}")
page = page.replace("__FONTFACE__", "\n".join(faces))
open(OUT, "w", encoding="utf-8").write(page)
print("written", OUT, len(page))
print("HEADLINE", HEADLINE)
for k in ["approve", "disapprove", "dem", "ppp", "rebuild", "reform", "progressive", "none"]:
    print(k, f1(first(k)[0]), "->", f1(last(k)[0]), last(k)[1].date())
print("cross", cross, "mrows", mrows)
print("house app", [(a, round(b, 1), c) for a, b, c in hitems])
