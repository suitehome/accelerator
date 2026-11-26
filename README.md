# INT8 MLP 추론 가속기 (모터 베어링 고장진단)

제2회 연세대 지능형반도체 프로젝트 경진대회 출품작. 병렬도 P를 파라미터로 둔 Verilog INT8 MLP 가속기와, 이와 비트 단위로 일치하는 Python 정수 골든 모델.

## 디렉터리

```
golden/qmlp.py          정수 골든 모델, 재양자화, ROM(hex) 내보내기, 예상 사이클 수
golden/random_model.py  검증용 랜덤 INT8 모델·입력 생성 (학습 모델 없이 RTL 검증)
rtl/mlp_accel.v         가속기 RTL (재양자화 5단 파이프라인, 현재 설계)
rtl/mlp_accel_comb.v    개선 전 RTL (재양자화 1사이클, 비교용)
synth/run_vivado.py     Vivado 합성·배치배선 일괄 실행
tb/tb_mlp.v             테스트벤치 (결과·뉴런별 trace 파일 출력)
run_sim.py              골든 모델 vs RTL 자동 비교
data/download_paderborn.py   Paderborn 데이터 다운로드·압축 해제
pipeline/paderborn.py   베어링 코드·라벨·분할, .mat 로더, BPFO/BPFI 계산
pipeline/features_vib.py  진동 포락선 특징 32개 (기본)
pipeline/features.py    전류 측파대 특징 32개 (--sensor current, 비교용)
pipeline/diagnose.py    파일별 전류·진동 포락선 스펙트럼 요약 (원인 분석용)
pipeline/mlp_np.py      NumPy MLP (PyTorch 불필요)
pipeline/quantize.py    INT8 PTQ, 벡터화 정수 추론(비트폭 스윕용)
pipeline/train.py       특징 → 학습 → 양자화 → model_q.json / test_inputs.json
pipeline/make_synthetic.py  파이프라인 단위 테스트용 합성 데이터
```

## 실행

필요: Python 3 + NumPy + SciPy, Icarus Verilog (`sudo apt install iverilog`, Windows는 bleyer.org/icarus 설치본)

```bash
python3 run_sim.py                     # 랜덤 모델, P = 1 4 16, 200 샘플
python3 run_sim.py --pow2 --n 500      # 정확히 .5인 반올림 경우를 대량으로 만드는 테스트
python3 run_sim.py --P 2 3 8 --seed 7  # 출력 수가 P로 나누어떨어지지 않는 경우
```

### 실제 데이터로 학습 → RTL 검증

```bash
python3 data/download_paderborn.py            # 정상 6 + 실제 손상 11개 베어링 (수 GB, 7-Zip/unar 필요)
python3 pipeline/train.py --data C:\paderborn        # 진동, 정상/손상 이진 (기본)
python3 run_sim.py --model results/model_q.json --inputs results/test_inputs.json \
                   --labels results/test_labels.json --P 1 4 16
```

`train.py` 옵션: `--sensor current|vibration`, `--task binary|3class`, `--conditions all`(4개 운전 조건), `--artificial`(인공 손상 베어링을 학습에 추가), `--hidden 16`, `--poles 4`(극쌍수 직접 지정).
결과는 `results/report.json`에 float·INT8 정확도, 파일 단위 다수결 정확도, 혼동행렬, 테스트 베어링별 정확도, 8/6/4 bit 스윕으로 저장된다.

불일치가 나면 첫 번째로 틀린 뉴런의 (샘플, 레이어, 출력 번호, v, y)를 골든과 RTL 양쪽 값으로 출력한다.

## 구조

- **출력 고정(output-stationary) MAC 배열 P개**를 FSM이 레이어마다 재사용한다.
- MAC 한 사이클에 입력 activation 1개를 모든 lane에 브로드캐스트하고, 가중치 ROM에서 8P bit 와이드 워드 1개(P개 가중치)를 읽는다. 가중치는 전부 온칩이므로 대역폭 병목이 없다.
- 한 그룹의 누산이 끝나면 P개 누산값을 사이클마다 하나씩 **재양자화 유닛 1개**(5단 파이프라인)에 넣는다(면적 ↔ 지연 절충). 레이어 경계에서만 4사이클 drain.
- 은닉층은 ReLU + [0, 127] 포화, 마지막 층은 [-128, 127] 포화 후 argmax(동점이면 앞 인덱스).

재양자화 (골든 모델과 RTL 공통):

```
v = Σ W[o][i]·x[i] + b[o]                 (RTL: 33 bit, 오버플로 없음)
y = (v·M0 + 2^(n-1)) >>> n                 (RTL: 50 bit, 반올림 half-up, 산술 시프트)
y = clamp(y, 0 또는 -128, 127)
```

M0, n은 `quantize_multiplier(M)`이 오프라인에서 한 번 계산한다(2^14 ≤ M0 < 2^15). 추론 경로에는 실수 연산이 없다.

## FPGA 결과 (Vivado 2026.1, Artix-7 xc7a35tcpg236-1, 32-16-2 모델)

| P | LUT | FF | DSP | Fmax | 사이클 | 지연 | 에너지/추론 |
|---|---|---|---|---|---|---|---|
| 1 | 475 | 151 | 2 | 94.6 MHz | 573 | 6.06 µs | 63 nJ |
| 4 | 834 | 243 | 2 | 98.9 MHz | 173 | 1.75 µs | 36 nJ |
| 16 | 2,053 | 621 | 2 | 90.7 MHz | 77 | 0.85 µs | 44 nJ |

`rtl/mlp_accel.v`(재양자화 5단 파이프라인) 기준. 개선 전 `rtl/mlp_accel_comb.v`(재양자화 1사이클)는 Fmax 55 MHz였다(`results/synth_summary_comb.csv`).
사이클 수: Σ_l(⌈O_l/P⌉·I_l + O_l) + 4·L + O_last + 1 (`golden/qmlp.py: cycles_expected`).

## 실제 데이터 결과 (Paderborn, 1500 rpm 기준 조건)

평가는 세 가지로 나눠 보고한다. 설계 선택(센서, 대역, 특징)은 1번으로만 정했고 테스트 베어링은 마지막에 한 번만 봤다.

| | 학습 베어링 LOBO | 테스트 베어링 6개 | 무작위 창 분할 (누수, 무효) |
|---|---|---|---|
| 진동 포락선, 이진 | 0.86 | **0.834** (INT8, float 대비 −0.06%p) | 0.97 |
| 진동 포락선, 3클래스 | 0.63 | – | – |
| 상전류 측파대, 3클래스 | – | 0.18 | 0.82 (파일 단위) |
| 상전류 전체 스펙트럼, 이진 | 0.59 (17개 전체 LOBO) | – | – |

- **상전류에는 이 데이터의 실제 손상 베어링 결함 신호가 없다.** 측파대 위치의 레벨이 정상·외륜·내륜 베어링 모두 같고(±1 dB), 무작위 분할 82% 대 베어링 분리 22%의 차이는 모델이 결함이 아니라 개별 베어링/측정 세션을 구분했다는 뜻이다.
- **진동 포락선**에서는 외륜 손상 KA04·KA16이 BPFO(약 78 Hz)에서 +21~23 dB로 뚜렷하다.
- 테스트 오분류는 거의 전부 **정상 베어링 K005**(1201창 중 596창, 손상으로 오판). 손상 베어링 2400창 중 놓친 것은 2개. 학습 LOBO에서도 정상 K004가 같은 양상이다.
- RTL: 테스트 입력 3601개, P = 1·4·16 모두 bit-exact, RTL 정확도 0.834 (573 / 173 / 77 사이클).

## 데이터 파이프라인

- **분할:** 베어링 단위. 학습 K001–K004, KA04·KA15·KA22, KI04·KI14·KI16·KI17 / 테스트 K005·K006, KA16·KA30, KI18·KI21 (실제 손상, 복합 손상 KB는 제외). 같은 베어링의 창이 양쪽에 섞이지 않는다(코드에서 assert).
- **스펙트럼:** 두 상전류의 Hann 창 FFT(L = 16384, Δf = 3.9 Hz, 50% 겹침) 파워 평균 → 기본파 대비 dB.
- **f_e 추정:** 20–400 Hz 최대 피크 + 포물선 보간. 극쌍수 p = round(f_e / f_r).
- **특징 선정 (중요):** 모든 측파대 주파수가 f_r에 비례하므로 f_r 배수(order)로 정의한다. 후보 |h·f_e ± k·f_c|, |h·f_e ± k·BPFI ± f_r| (h = 1, 3, 5)를 우선순위대로 보면서
  - 기본파 홀수 고조파·DC에서 0.6 order 이내인 선은 제외(큰 고조파의 누설이 지배)
  - 이미 고른 선과 0.6 order 이내로 겹치는 선은 제외
  하여 32개를 고른다. 두 번째 규칙이 필요한 이유: BPFO + BPFI = N·f_r 이므로 볼 8개, p = 4이면 BPFO + BPFI = 2·f_e 가 되어 **f_e − BPFO 와 BPFI − f_e 가 같은 주파수(1500 rpm에서 23.6 Hz)** 가 된다. 이 선은 외륜·내륜을 구분하지 못한다. 겹침은 속도와 무관하고 p와 베어링 형상만으로 정해진다.
- **양자화:** 대칭 per-tensor PTQ, 활성값 스케일은 학습셋 99.9 백분위. `train.py`가 벡터화 정수 추론과 `golden/qmlp.py` 결과가 테스트셋 전체에서 같은지 assert한다.

### 합성 데이터 단위 테스트

`pipeline/make_synthetic.py`는 Paderborn과 같은 폴더·파일명·.mat 구조로 측파대를 넣은 전류를 만든다. 로더 → 특징 → 학습 → 양자화 → RTL까지 끝까지 도는지 확인하는 용도이며, 여기서 나온 정확도(합성 데이터에서는 100%)는 **진단 성능의 근거가 아니다.**


## 다음 단계

자세한 설명, 수정 시 같이 고쳐야 할 곳, 남은 과제(MAC 경로 파이프라인화, 비트폭 8·6·4 설계점, 정상 베어링 오경보)는 인수인계 가이드(`00_인수인계_가이드.pdf`)를 본다.
