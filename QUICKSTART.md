# 🚀 Quick Start Guide

## 빠른 실행

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 통합 테스트 (선택사항)
python test_integration.py

# 3. Streamlit 앱 실행
streamlit run app.py
```

## 브라우저에서 확인

- 자동으로 `http://localhost:8501` 열림
- 좌측 사이드바에서 설정 조정
- **Run Simulation** 클릭!

## 3분 만에 핵심 체험하기

### Step 1: 기본 설정으로 실행 (1분)
- 모든 기본값 유지
- "Run Simulation" 클릭
- Overview 탭에서 온도 분포 확인

### Step 2: Training 과정 이해 (1분)
- "Training Comparison" 탭 이동
- **PINN Loss Breakdown** 그래프 주목!
  - PDE Loss가 0으로 수렴 = 물리 법칙 학습!

### Step 3: 일반화 테스트 (1분)
- "Generalization Test" 탭 이동
- **핵심:** 100mm로 학습 → 200mm 예측
- Pure AI: 실패 (외삽 불가)
- PINN: 성공! (물리 법칙 학습)

## 핵심만 빠르게!

### PINN이 뭔가요?
**Physics-Informed Neural Network**
- 일반 AI: 데이터만 학습
- PINN: **물리 법칙을 학습**

### 왜 중요한가요?
1. ✅ **적은 데이터로 학습**
2. ✅ **일반화 능력 우수**
3. ✅ **물리적으로 타당한 예측**

### 어떻게 동작하나요?
```
Loss = 데이터 피팅 + 물리법칙 만족도
     = MSE(예측, 실제) + MSE(PDE 잔차, 0)
```

## 추천 실험

### 실험 1: 데이터 부족 상황
- Measurement Interval: 10 → 50
- Pure AI 성능 급격히 하락
- PINN은 비교적 안정적

### 실험 2: 다른 물질
- Thermal Diffusivity 변경
- 1e-5 (낮음): 천천히 퍼짐
- 1e-3 (높음): 빠르게 퍼짐

### 실험 3: 긴 시뮬레이션
- Simulation Time: 100s → 300s
- 장기 예측에서도 PINN 우수

## 문제 해결

### 학습이 너무 느려요
- Epochs 줄이기 (Pure AI: 1000→500, PINN: 3000→1500)
- 또는 nx 줄이기 (100→50)

### Loss가 수렴 안 해요
- Learning rate 조정
- Epochs 늘리기
- 또는 네트워크 크기 조정

### Streamlit 실행 안 돼요
```bash
# Streamlit 재설치
pip uninstall streamlit
pip install streamlit
```

## 다음 단계

1. 📖 **README.md** 읽기 - 상세 설명
2. 🧪 **test_integration.py** 코드 분석
3. 🔬 **models/pinn.py** 구조 이해
4. 🎨 **utils/visualization.py** 커스터마이징

## 핵심 파일

```
app.py              ← Streamlit UI (여기서 시작!)
models/pinn.py      ← PINN 핵심 구현
models/pure_ai.py   ← 순수 AI 구현
utils/physics.py    ← CAE solver
```

## 즐기세요! 🎉

**질문이나 피드백은 언제든 환영합니다!**

---

*"AI가 물리 법칙을 학습한다" - PINN의 핵심*
