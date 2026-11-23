"""Keyword topic classifier for Korean financial news (FinSent), version 2.

Design rules
- Ten content categories plus 'other', fixed before estimation.
- Leader titles and person-type words (회장, 부회장, 대표, 총수, 오너, CEO ...) are NOT keywords, so the
  topic label does not mechanically encode leader framing.
- Score = 2 x hits in the headline + 1 x hits in the first 120 characters of the body text.
  Highest score wins; ties go to the earlier category below; no hits -> 'other'.
"""
import re
import numpy as np
import pandas as pd

TOPICS = {
    "legal_regulatory": [r"검찰", r"기소", r"재판", r"판결", r"선고", r"소송", r"과징금", r"공정위", r"공정거래", r"제재",
                          r"수사", r"압수수색", r"혐의", r"구속", r"벌금", r"금감원", r"금융감독", r"징계", r"고발",
                          r"고소", r"위반", r"횡령", r"배임", r"담합", r"리콜", r"사고", r"유출", r"해킹", r"중대재해",
                          r"세무조사", r"법원", r"대법", r"항소", r"상고", r"무죄", r"유죄", r"사면", r"특사", r"가석방",
                          r"경찰", r"규제", r"부당", r"사망", r"화재", r"폭발", r"장애", r"유심"],
    "mna_deals": [r"인수", r"합병", r"M&A", r"매각", r"지분", r"합작", r"상장", r"IPO", r"블록딜", r"매입",
                  r"투자 유치", r"투자유치", r"시리즈[ ]?[A-D]", r"물적분할", r"인적분할"],
    "earnings": [r"실적", r"영업이익", r"영업익", r"순이익", r"매출", r"적자", r"흑자", r"잠정", r"영업손실", r"순손실",
                 r"어닝"],
    "governance_ownership": [r"지배구조", r"승계", r"지주사", r"지주회사", r"경영권", r"상속", r"주총", r"주주총회",
                             r"이사회", r"사외이사", r"배당", r"주주환원", r"밸류업", r"증여", r"등기임원", r"준법"],
    "personnel": [r"\[인사\]", r"임원인사", r"정기 인사", r"인사 단행", r"사장단", r"선임", r"임명", r"승진", r"취임",
                  r"사임", r"퇴임", r"내정", r"연임", r"조직개편", r"신임", r"새 대표", r"혁신팀"],
    "contracts_orders": [r"수주", r"계약", r"MOU", r"업무협약", r"협약", r"맞손", r"파트너십", r"제휴", r"손잡",
                         r"협업", r"협력", r"체결"],
    "investment_capex": [r"투자(?!자|의견|유치|증권|설명|전문|귀재)", r"증설", r"공장", r"설비", r"착공", r"준공",
                         r"생산능력", r"캐파", r"거점", r"건립", r"고로", r"데이터 ?센터", r"건설"],
    "products_tech": [r"출시", r"신제품", r"선보", r"론칭", r"런칭", r"신차", r"신약", r"임상", r"허가", r"특허",
                      r"사전계약", r"개발", r"도입", r"앱", r"챗봇", r"체험", r"광고", r"판매"],
    "stock_market": [r"주가", r"급등", r"급락", r"강세", r"약세", r"시총", r"시가총액", r"코스피", r"코스닥",
                     r"외국인", r"순매수", r"순매도", r"목표주가", r"목표가", r"투자의견", r"증권가", r"신고가",
                     r"특징주", r"증시", r"주간전망", r"종목", r"주식"],
    "csr_esg_events": [r"봉사", r"기부", r"후원", r"사회공헌", r"ESG", r"탄소중립", r"친환경", r"상생", r"나눔",
                       r"캠페인", r"시상", r"수상", r"선정", r"기념", r"전시", r"박람회", r"초청", r"협력사",
                       r"조기 지급", r"헌혈", r"장애인", r"개소", r"RE100"],
}
TOPIC_NAMES = list(TOPICS) + ["other"]
_PATTERNS = {k: re.compile("|".join(v)) for k, v in TOPICS.items()}
LEAD_CHARS = 120


def classify(title: pd.Series, lead: pd.Series) -> pd.Series:
    title = title.fillna("").astype(str)
    lead = lead.fillna("").astype(str).str.slice(0, LEAD_CHARS)
    scores = np.column_stack([
        2 * title.str.count(p).to_numpy() + lead.str.count(p).to_numpy() for p in _PATTERNS.values()
    ])
    out = np.array(list(TOPICS))[scores.argmax(axis=1)]
    out[scores.max(axis=1) == 0] = "other"
    return pd.Series(out, index=title.index, name="topic")
