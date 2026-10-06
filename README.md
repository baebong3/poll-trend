# 여론조사 종합 추세

(주)서던포스트가 중앙선거여론조사심의위원회 등록 전국 여론조사(대통령 국정수행 평가, 정당지지도)를 종합한 추세 페이지.
페이지 수치는 서던포스트가 직접 조사한 결과가 아니다.

## 구성

| 파일 | 역할 |
|---|---|
| `data/polls.csv` | 정본 자료(1행 = 1개 조사) |
| `build.py` | `polls.csv` → `index.html` (JS 없이 보이는 단일 파일, 차트는 SVG) |
| `scripts/collect.py` | Claude API(웹 검색·웹 가져오기)로 새 조사를 찾아 `polls.csv`에 추가 |
| `scripts/check_html.py` | 빌드 결과에 nan·줄표가 없는지 점검 |
| `.github/workflows/update.yml` | 매일 13:10 KST 수집 → 빌드 → 커밋 → GitHub Pages 배포 |

## 자동 수집 안전장치

- 출처 URL은 그 실행에서 실제로 본문을 연 기사만 인정
- 모델이 낸 숫자도 기사 본문에 그 숫자가 있는지 다시 대조, 없으면 빈칸
- 같은 조사기관·방법·조사기간 행은 새로 넣지 않고 빈칸만 보충
- nesdc.go.kr 접근 차단
- 자동으로 들어간 행은 `note` 끝에 `자동수집` 표시
- 실행마다 Actions 요약에 신규·제외·빈칸 처리 내역, `collect-log` 아티팩트에 로그(30일 보관)

## 수동 수정

`data/polls.csv`를 고쳐 main에 커밋하면 수집 없이 페이지만 다시 빌드·배포된다.

## 설정

- Secrets `ANTHROPIC_API_KEY` (필수)
- Variables `CLAUDE_MODEL` (선택, 기본 `claude-sonnet-5-5`)
- Settings > Pages > Source: GitHub Actions
