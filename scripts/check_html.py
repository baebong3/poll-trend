# -*- coding: utf-8 -*-
"""빌드 결과 점검: 글꼴 데이터(data: URI)를 뺀 본문에 nan·NaN·줄표가 없어야 한다."""
import re, sys
p = sys.argv[1] if len(sys.argv) > 1 else "site/index.html"
t = re.sub(r"data:[^\"')]+", "", open(p, encoding="utf-8").read())
bad = [(m.group(), t[max(0, m.start() - 40): m.end() + 40]) for m in re.finditer(r"nan|NaN|—|–", t)]
if bad:
    for b in bad[:10]:
        print("점검 실패:", b)
    sys.exit(1)
print("점검 통과:", p)
