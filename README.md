# 개인연금(IRP) 월간 투자 계획 agent

매달 들어오는 연금 납입금을 **장기 60% / 단기 40%**로 나누고, 경제 상황과 ETF 추세를 바탕으로
"이번 달에 어떤 ETF를 몇 주 살지"와 "그 ETF가 무엇인지"를 정리한 리포트를 만듭니다.

- 계좌: IRP 2개(본인 신한투자증권 월 40만 원, 남편 우리투자증권 월 20만 원)
- 목표: 만 65세 연금 수령, 최대 손실 -10% 이내, 한쪽에 쏠리지 않는 분산
- IRP 규칙 반영: 위험자산 70% 한도, 레버리지·인버스·선물형 ETF 제외
- 데이터: API 키 없이 FRED(미국 경제지표), 네이버 금융(ETF 시세), Google News(뉴스)

> 공부용 참고 자료이며 투자 권유가 아닙니다.

## 실행

```bash
pip install -r requirements.txt
python -m pension_agent --assume-last-plan   # reports/YYYY-MM.md 생성
python -m pytest -q                          # 테스트(네트워크 불필요)
```

| 옵션 | 설명 |
|---|---|
| `--assume-last-plan` | 지난 리포트의 매수 계획대로 샀다고 가정해 `config/holdings.yaml` 갱신 |
| `--month 2026-12` | 리포트 파일명 지정 |
| `--no-news` | 뉴스 수집 생략 |

필요한 네트워크 도메인: `fred.stlouisfed.org`, `fchart.stock.naver.com`, `news.google.com`

## 리포트 구성

1. 이번 달 할 일: 계좌별 매수 종목과 수량
2. 경제 상황 점검: 지표별 신호와 경제 국면
3. 목표 포트폴리오와 위험 점검(과거 시뮬레이션 최대 손실, IRP 한도)
4. 장기 바구니: 종목별 "이게 뭔가요? / 왜 담나요? / 공부 포인트"
5. 단기 바구니: 선정 이유와 후보 순위표(제외 사유 포함)
6. 공부할 후보 ETF 전체 목록
7. 최근 2주 주요 뉴스
8. Claude 해설(월간 자동 실행 때 작성)

## 설정 파일

- `config/profile.yaml`: 납입액, 위험 한도, 장단기 비중, 월 최대 매수 종목 수
- `config/etf_universe.yaml`: 후보 ETF와 설명
- `config/holdings.yaml`: 보유 수량 장부(예시: `config/holdings.example.yaml`)

자세한 로직은 [docs/설계.md](docs/설계.md), Claude 실행 절차는 [CLAUDE.md](CLAUDE.md)에 있습니다.
