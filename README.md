# IQ to 64-bit Hex Converter (Q1.15 Fixed-Point)

계측 장비에서 출력된 I/Q float 데이터를 장비 규격에 맞는  
Q1.15 고정소수점 + 64-bit packed hex word로 변환하는 Python 스크립트입니다.

학기 중 국가근로 업무로 관측 데이터를 후처리하면서,
반복적으로 수작업으로 하던 변환 과정을 자동화 스크립트로 구현했습니다.

---

## 기능 요약

- CSV에서 I/Q 데이터 1296쌍을 자동 파싱  
  - 단일 컬럼 (I,Q 번갈아) / 다중 컬럼(I열, Q열) 모두 대응
- float → Q1.15(int16) 고정소수점 변환 및 클리핑
- (I0, Q0, I1, Q1) → 64-bit word 패킹  
  `word = I0<<48 | Q0<<32 | I1<<16 | Q1`
- 36×36 스크린샷 형태의 데이터를 BFM1/2/3 블록으로 분할
- 템플릿 CSV(qsfp) 상단 헤더를 복사하고,
  템플릿의 컬럼 수(18/36)에 따라 포맷을 자동으로 맞춤
- 최종적으로 장비 입력 규격에 맞는 CSV 파일을 생성

---

## 사용 방법

1. Python 3.x 환경에서 필요한 라이브러리를 설치합니다.

pip install -r requirements.txt

2. 템플릿 CSV 파일과 입력 CSV 파일 경로를 스크립트 상단에서 설정합니다.

3. 아래와 같이 실행합니다.

python convert_bfm.py --template qsfp.csv header_lines 9


template : 헤더로 사용할 템플릿 CSV 파일

header_lines : 템플릿에서 상단 몇 줄을 헤더로 복사할지