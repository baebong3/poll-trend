# -*- coding: utf-8 -*-
"""
새로 공표된 전국 여론조사를 Claude API(웹 검색·웹 가져오기)로 찾아 polls.csv에 더한다.

원칙
- 원문에서 직접 확인한 숫자만 넣는다. 모델이 돌려준 숫자도 실제로 가져온 기사 본문에
  그 숫자가 있는지 다시 대조하고, 없으면 그 칸은 빈칸으로 둔다.
- 출처 URL은 이번 실행에서 실제로 가져온(web_fetch) 페이지여야 한다.
- 같은 pollster·method·field_start·field_end 행이 있으면 새로 넣지 않고 빈칸만 채운다.
- 중앙선거여론조사심의위원회(nesdc.go.kr)는 접근하지 않는다.

실행: python scripts/collect.py data/polls.csv
환경변수: ANTHROPIC_API_KEY(필수), CLAUDE_MODEL(기본 claude-sonnet-5-5)
"""
import csv, json, os, re, sys, time, datetime as dt, urllib.request, urllib.error

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/polls.csv"
MODEL = os.environ.get("CLAUDE_MODEL") or "claude-sonnet-5-5"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
KST = dt.timezone(dt.timedelta(hours=9))
TODAY = dt.datetime.now(KST).date()

COLS = ["pollster", "sponsor", "method", "sample_n", "field_start", "field_end", "moe", "response_rate",
        "approve", "disapprove", "dem", "ppp", "rebuild", "reform", "progressive", "none", "source_url", "note"]
PCT = ["approve", "disapprove", "dem", "ppp", "rebuild", "reform", "progressive", "none"]
NUM = ["sample_n", "moe", "response_rate"] + PCT
METHODS = {"전화면접", "무선ARS", "혼합", "웹"}
CANON = [(("NBS", "전국지표"), "NBS(엠브레인퍼블릭·케이스탯리서치·코리아리서치·한국리서치)"),
         (("KSOI", "한국사회여론연구소"), "한국사회여론연구소(KSOI)")]
BLOCKED = ["nesdc.go.kr"]


# ---------------------------------------------------------------- CSV
def load_rows():
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def save_rows(rows):
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})

def key(r):
    return (r["pollster"].strip(), r["method"].strip(), r["field_start"].strip(), r["field_end"].strip())


# ---------------------------------------------------------------- API
def call(messages, system, tools):
    body = json.dumps({"model": MODEL, "max_tokens": 16000, "system": system,
                       "messages": messages, "tools": tools}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST", headers={
        "x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(20 * (attempt + 1)); continue
            raise SystemExit(f"API 오류 {e.code}: {msg}")
    raise SystemExit("API 재시도 초과")

def run_agent(system, prompt):
    tools = [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 20, "blocked_domains": BLOCKED,
         "user_location": {"type": "approximate", "country": "KR", "timezone": "Asia/Seoul"}},
        {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 25, "blocked_domains": BLOCKED,
         "max_content_tokens": 15000},
    ]
    messages = [{"role": "user", "content": prompt}]
    blocks, usage = [], {"input_tokens": 0, "output_tokens": 0}
    for _ in range(10):
        resp = call(messages, system, tools)
        for k in usage:
            usage[k] += resp.get("usage", {}).get(k, 0) or 0
        blocks += resp["content"]
        if resp.get("stop_reason") == "pause_turn":
            messages = messages + [{"role": "assistant", "content": resp["content"]}]
            continue
        break
    return blocks, usage


# ---------------------------------------------------------------- 대조
def norm_url(u):
    u = (u or "").strip()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^(www\.|m\.)", "", u)
    return u.rstrip("/")

def fetched_pages(blocks):
    """web_fetch로 실제 가져온 페이지: {정규화 URL: 본문}"""
    pages = {}
    for b in blocks:
        if b.get("type") != "web_fetch_tool_result":
            continue
        c = b.get("content") or {}
        if c.get("type") != "web_fetch_result":
            continue
        doc = (c.get("content") or {}).get("source") or {}
        text = doc.get("data", "") if doc.get("type") == "text" else ""
        text += " " + ((c.get("content") or {}).get("title") or "")
        pages[norm_url(c.get("url"))] = text
    return pages

def num_in(text, v, field):
    """기사 본문에 그 숫자가 실제로 있는지(37.4 / 38% / 2,515명 등)"""
    if field == "sample_n":
        n = int(round(v))
        return bool(re.search(rf"(?<![\d.,]){n:,}(?![\d])|(?<![\d.,]){n}(?![\d])", text))
    s = f"{v:.1f}"
    pats = [re.escape(s)]
    if s.endswith(".0"):
        pats.append(rf"{int(v)}(?:\s*%|\s*퍼센트|\s*％)")
    return any(re.search(rf"(?<![\d.]){p}(?![\d])", text) for p in pats)

def clean_number(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except ValueError:
        return None

def fmt(field, v):
    if v is None:
        return ""
    return str(int(round(v))) if field == "sample_n" else f"{v:.1f}"

def canon_pollster(p, known):
    p = (p or "").strip()
    for keys, c in CANON:
        if any(k in p for k in keys):
            return c
    for k in known:
        if p == k or p.replace(" ", "") == k.replace(" ", ""):
            return k
    return p

def valid_date(s):
    try:
        return dt.date.fromisoformat(str(s).strip())
    except ValueError:
        return None


def verify(item, pages, log, label):
    """숫자 칸마다 근거 페이지 본문 대조. 확인 안 된 칸은 None."""
    urls = [item.get("source_url", "")] + list(item.get("evidence_urls") or [])
    texts = [pages[norm_url(u)] for u in urls if norm_url(u) in pages]
    out = {}
    for f in NUM:
        v = clean_number(item.get(f))
        if v is None:
            continue
        if f in PCT and not (0 <= v <= 100):
            log.append(f"{label}: {f}={v} 범위 밖, 제외"); continue
        if f == "sample_n" and not (300 <= v <= 20000):
            log.append(f"{label}: 표본 {v} 범위 밖, 제외"); continue
        if any(num_in(t, v, f) for t in texts):
            out[f] = v
        else:
            log.append(f"{label}: {f}={v} 원문에서 확인 안 됨, 빈칸 처리")
    return out


# ---------------------------------------------------------------- 메인
SYSTEM = """너는 한국 전국 단위 여론조사 자료를 정리하는 조사원이다. 대통령 국정수행 평가와 정당지지도 수치를 기사 원문에서 그대로 옮긴다.
규칙
- 중앙선거여론조사심의위원회(nesdc.go.kr)는 열지 않는다. "중앙선거여론조사심의위원회 홈페이지 참조"라고 밝힌 언론 보도와 조사기관 공개 자료(gallup.co.kr 등)를 web_search로 찾고 web_fetch로 본문을 연다.
- 숫자는 반드시 web_fetch로 연 본문에서 직접 확인한 것만 쓴다. 검색 결과 제목·요약에만 보이는 숫자는 쓰지 않는다. 없는 값은 null로 두고 추정하지 않는다.
- 리얼미터(에너지경제신문 의뢰)는 국정수행과 정당지지도가 표본·기간이 다른 별도 조사이므로 각각 한 행이다(국정수행 행은 정당 칸 null, 정당 행은 긍정·부정 null).
- 여론조사꽃은 전화면접판과 ARS판을 각각 한 행으로 쓴다.
- method는 전화면접/무선ARS/혼합/웹 중 하나. 유선 일부가 섞인 ARS도 무선ARS로 쓰고 note에 비율을 적는다.
- 대상: 한국갤럽, NBS 전국지표조사, 리얼미터, 한국사회여론연구소(KSOI), 조원씨앤아이, 여론조사꽃, 미디어토마토, 에이스리서치, 리서치뷰, 한길리서치 및 그 밖의 전국 조사. 지역 조사·가상대결만 있는 조사는 제외.
- 마지막 답변은 설명 없이 ```json 코드블록 하나만 낸다."""

def build_prompt(rows):
    last_end = max(r["field_end"] for r in rows)
    since = (valid_date(last_end) - dt.timedelta(days=21)).isoformat()
    recent = [r for r in rows if r["field_end"] >= since]
    have = "\n".join(f"- {r['pollster']} | {r['method']} | {r['field_start']}~{r['field_end']}"
                     + (" | 빈칸: " + ",".join(c for c in PCT + ['moe', 'response_rate'] if not r.get(c)) if any(not r.get(c) for c in PCT) else "")
                     for r in recent)
    pollsters = sorted({r["pollster"] for r in rows})
    return f"""오늘은 {TODAY.isoformat()}(한국시간)이다. 지금 자료의 가장 늦은 조사 종료일은 {last_end}이다.
{last_end} 이후 공표된 전국 여론조사(대통령 국정수행 평가, 정당지지도)를 찾아라.

이미 들어 있는 최근 조사(같은 조사기관·방법·기간이면 새 행으로 넣지 말 것. 빈칸이 있으면 새 보도로 확인된 값만 fills로):
{have}

pollster 표기는 기존 표기를 그대로 쓴다: {", ".join(pollsters)}

출력 형식(JSON):
{{
  "new_rows": [{{"pollster": "", "sponsor": "", "method": "", "sample_n": 0, "field_start": "YYYY-MM-DD", "field_end": "YYYY-MM-DD",
     "moe": null, "response_rate": null, "approve": null, "disapprove": null, "dem": null, "ppp": null, "rebuild": null,
     "reform": null, "progressive": null, "none": null, "source_url": "web_fetch로 연 기사 URL",
     "evidence_urls": ["값 일부를 다른 기사에서 확인했으면 그 URL"], "note": "공표일·특이사항(예: 국정수행; 2026-10-05 공표)"}}],
  "fills": [{{"pollster": "", "method": "", "field_start": "", "field_end": "", "values": {{"칸이름": 값}}, "source_url": ""}}],
  "checked": "확인한 기관과 결과 한두 줄"
}}
새 조사가 없으면 new_rows를 빈 배열로 둔다."""

def parse_json(blocks):
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)
    raw = m[-1] if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(raw)
    except Exception:
        raise SystemExit("모델 응답에서 JSON을 읽지 못함:\n" + text[-2000:])


def apply(rows, data, pages, log):
    known = sorted({r["pollster"] for r in rows})
    index = {key(r): r for r in rows}
    last_end = valid_date(max(r["field_end"] for r in rows))
    added, filled = [], []

    for it in data.get("new_rows") or []:
        p = canon_pollster(it.get("pollster"), known)
        m = (it.get("method") or "").replace(" ", "")
        m = "무선ARS" if m in ("ARS", "자동응답") else m
        fs, fe = valid_date(it.get("field_start", "")), valid_date(it.get("field_end", ""))
        label = f"{p} {m} {it.get('field_start')}~{it.get('field_end')}"
        if not p or m not in METHODS or not fs or not fe or fs > fe or fe > TODAY or (fe - fs).days > 14:
            log.append(f"{label}: 기본 항목 오류, 제외"); continue
        if fe < last_end - dt.timedelta(days=21):
            log.append(f"{label}: 너무 오래된 조사, 제외"); continue
        if norm_url(it.get("source_url")) not in pages:
            log.append(f"{label}: 출처 URL을 실제로 열지 않음, 제외"); continue
        k = (p, m, fs.isoformat(), fe.isoformat())
        vals = verify(it, pages, log, label)
        if k in index:  # 이미 있으면 빈칸만
            r = index[k]; ch = [f for f, v in vals.items() if not r.get(f)]
            for f in ch:
                r[f] = fmt(f, vals[f])
            if ch:
                filled.append(f"{label}: {','.join(ch)}")
            continue
        if "approve" not in vals and "dem" not in vals:
            log.append(f"{label}: 국정·정당 수치가 확인되지 않아 제외"); continue
        if vals.get("approve", 0) + vals.get("disapprove", 0) > 100.5:
            log.append(f"{label}: 긍정+부정 100 초과, 제외"); continue
        if sum(vals.get(c, 0) for c in PCT[2:]) > 100.5:
            log.append(f"{label}: 정당 합 100 초과, 제외"); continue
        note = (it.get("note") or "").strip()
        r = {"pollster": p, "sponsor": (it.get("sponsor") or "").strip(), "method": m,
             "field_start": fs.isoformat(), "field_end": fe.isoformat(),
             "source_url": it["source_url"].strip(), "note": (note + "; " if note else "") + "자동수집"}
        for f in NUM:
            r[f] = fmt(f, vals.get(f))
        rows.append(r); index[k] = r; added.append(r)

    for it in data.get("fills") or []:
        p = canon_pollster(it.get("pollster"), known)
        k = (p, (it.get("method") or "").strip(), str(it.get("field_start", "")).strip(), str(it.get("field_end", "")).strip())
        r = index.get(k)
        if not r or norm_url(it.get("source_url")) not in pages:
            continue
        label = f"{p} {k[1]} {k[2]}~{k[3]} 보충"
        vals = verify({**(it.get("values") or {}), "source_url": it.get("source_url")}, pages, log, label)
        ch = [f for f, v in vals.items() if not r.get(f)]
        for f in ch:
            r[f] = fmt(f, vals[f])
        if ch:
            r["note"] = (r.get("note") or "") + f"; {','.join(ch)} 보충은 {it['source_url']}"
            filled.append(f"{label}: {','.join(ch)}")
    return added, filled


def main():
    if not API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY가 없습니다(저장소 Settings > Secrets에 등록).")
    rows = load_rows()
    blocks, usage = run_agent(SYSTEM, build_prompt(rows))
    pages = fetched_pages(blocks)
    data = parse_json(blocks)
    log = []
    added, filled = apply(rows, data, pages, log)
    if added or filled:
        save_rows(rows)

    summary = {"date": TODAY.isoformat(), "model": MODEL, "added": len(added), "filled": filled,
               "rows": [f"{r['pollster']} {r['method']} {r['field_start']}~{r['field_end']}" for r in added],
               "rejected": log, "fetched_pages": len(pages), "usage": usage, "checked": data.get("checked", "")}
    json.dump(summary, open("collect_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"changed={'true' if (added or filled) else 'false'}\nadded={len(added)}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"## 여론조사 수집 {TODAY}\n\n- 신규 {len(added)}건, 보충 {len(filled)}건, 연 페이지 {len(pages)}개\n")
            for s in summary["rows"]:
                f.write(f"  - {s}\n")
            if log:
                f.write("\n### 제외·빈칸 처리\n" + "".join(f"- {s}\n" for s in log))
            f.write(f"\n토큰: 입력 {usage['input_tokens']:,} / 출력 {usage['output_tokens']:,}\n")


def fail_summary(msg):
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"## 여론조사 수집 실패 {TODAY}\n\n```\n{msg}\n```\n")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"::error title=수집 실패::{str(e.code)[:300]}")
            fail_summary(str(e.code))
        raise
