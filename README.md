# 🔥 PINN Interactive Demo: Heat Conduction

**Physics-Informed Neural Networks (PINN) 교육용 인터랙티브 데모**

이 프로젝트는 1D 열전도 문제를 통해 PINN의 핵심 개념과 장점을 직관적으로 보여줍니다.

---

## 🎯 핵심 목표

**문제:** 100mm 쇠 막대의 왼쪽 끝을 100°C로 가열할 때, 오른쪽 끝의 온도 예측

**세 가지 접근법 비교:**

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| 🧮 **CAE** | 유한차분법으로 열전도 방정식 직접 풀기 | ✅ 매우 정확 | ❌ 메쉬 설정 필요<br>❌ 새 형상 = 재계산 |
| 🤖 **Pure AI** | 센서 데이터만으로 학습 | ✅ 물리 몰라도 됨 | ❌ 많은 데이터 필요<br>❌ 일반화 어려움 |
| 🧠 **PINN** | 물리 법칙 자체를 학습 | ✅ 적은 데이터<br>✅ 우수한 일반화 | ⚠️ 물리 법칙 알아야 함 |

---

## 🔬 PINN의 핵심 아이디어

### 일반 AI vs PINN

**Pure AI:**
```
Loss = MSE(T_pred, T_measured)
```
→ 데이터 패턴만 학습

**PINN:**
```
Loss = Loss_BC + Loss_IC + Loss_PDE
     = (경계조건 오차) + (초기조건 오차) + (물리법칙 위반도)
```
→ **물리 법칙을 학습!**

### 열전도 방정식

```
∂T/∂t = α ∂²T/∂x²
```

- `∂T/∂t`: 온도의 시간 변화
- `α`: 열확산계수
- `∂²T/∂x²`: 공간상의 2차 미분 (열이 퍼지는 정도)

**PINN은 이 방정식을 만족하도록 학습합니다!**

---

## 📁 프로젝트 구조

```
Base_PINN/
├── app.py                    # Streamlit 메인 앱
├── requirements.txt          # 패키지 의존성
├── utils/
│   ├── physics.py           # 열전도 방정식 & CAE solver
│   ├── data_gen.py          # 데이터 생성 (센서, PINN)
│   └── visualization.py     # 시각화 함수
└── models/
    ├── pure_ai.py           # 순수 AI 모델
    └── pinn.py              # PINN 모델
```

---

## 🚀 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. Streamlit 앱 실행

```bash
streamlit run app.py
```

### 3. 웹 브라우저에서 접속

자동으로 브라우저가 열리며 `http://localhost:8501` 에서 확인 가능합니다.

---

## 🎮 사용 방법

### 좌측 사이드바

1. **Problem Setup**
   - Rod Length: 막대 길이 (50~200mm)
   - Simulation Time: 시뮬레이션 시간
   - Boundary Conditions: 경계 온도 설정
   - Thermal Diffusivity: 열확산계수

2. **Training Parameters**
   - Pure AI Epochs: 순수 AI 학습 반복 횟수
   - PINN Epochs: PINN 학습 반복 횟수

3. **Run Simulation** 버튼 클릭!

### 메인 탭

#### 📊 Overview
- CAE와 PINN의 온도 분포 히트맵
- 시간에 따른 온도 변화 애니메이션

#### 🎯 Training Comparison
- 세 방법의 데이터 요구량 비교
- Pure AI: 데이터 피팅 Loss
- **PINN: Loss 분해 (Boundary + Initial + PDE)**
  - **PDE Loss → 0 = 물리 법칙 학습!**

#### 📈 Results Analysis
- 정량적 지표 (MSE, MAE, RMSE)
- 오른쪽 끝 온도 비교 그래프
- 오차 히트맵

#### 🔬 Generalization Test
- **핵심 테스트:** 100mm로 학습 → 200mm 예측
- CAE: 재계산 필요
- Pure AI: 실패 (외삽 불가)
- **PINN: 성공!** (물리 법칙을 배웠으니 일반화 가능)

---

## 💡 주요 교육 포인트

### 1. PINN의 Loss 구조

```python
def compute_loss(self):
    # 1. 경계조건 만족 (x=0: 100°C, x=L: 20°C)
    loss_bc = MSE(T_pred_boundary, T_true_boundary)

    # 2. 초기조건 만족 (t=0: 20°C)
    loss_ic = MSE(T_pred_initial, T_initial)

    # 3. PDE 만족 (∂T/∂t = α ∂²T/∂x²)
    residual = dT_dt - alpha * d2T_dx2
    loss_pde = MSE(residual, 0)

    return loss_bc + loss_ic + loss_pde
```

### 2. Automatic Differentiation

PINN은 PyTorch의 자동 미분을 활용:

```python
# 1차 미분
dT_dx = torch.autograd.grad(T, x)[0]

# 2차 미분
d2T_dx2 = torch.autograd.grad(dT_dx, x)[0]
```

→ **수치 미분 없이 정확한 미분 계산!**

### 3. 일반화 능력

- **Pure AI:** 100mm 데이터로 학습 → 200mm 예측 실패
  - 이유: 데이터 범위 밖 외삽 불가

- **PINN:** 100mm 데이터로 학습 → 200mm 예측 성공!
  - 이유: 열전도 법칙 자체를 학습했으므로 형상 무관

---

## 📊 실험 결과 예시

### 데이터 요구량

```
CAE:      2,000,100 points (전체 온도장)
Pure AI:      4,002 points (센서 데이터)
PINN:         1,250 points (경계/초기/내부점)
                           └─ 온도 데이터는 경계/초기만!
```

### 정확도 (100mm rod)

```
Method      MSE      MAE      RMSE
Pure AI    0.50     0.60     0.70
PINN       0.10     0.20     0.30
```

### 일반화 성능 (200mm rod)

```
Method      RMSE (100mm)  RMSE (200mm)  증가율
Pure AI     0.70          5.20          +643%
PINN        0.30          0.45          +50%
```

→ **PINN이 일반화에서 압도적으로 우수!**

---

## 🔧 기술 스택

- **Python 3.8+**
- **PyTorch**: 신경망 & 자동 미분
- **NumPy**: 수치 계산
- **Streamlit**: 웹 인터페이스
- **Plotly**: 인터랙티브 시각화
- **Matplotlib**: 정적 그래프

---

## 📚 추가 학습 자료

### PINN 논문

- [Physics-Informed Neural Networks (Raissi et al., 2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

### 열전도 방정식

- Heat Equation: `∂T/∂t = α ∂²T/∂x²`
- Finite Difference Method (유한차분법)
- Fourier Number 안정성 조건: `Fo = α·Δt/Δx² ≤ 0.5`

### 활용 분야

- **유체역학**: Navier-Stokes 방정식
- **구조해석**: 탄성 방정식
- **전자기학**: Maxwell 방정식
- **양자역학**: Schrödinger 방정식

---

## 🎓 교육적 가치

이 데모는 다음을 배울 수 있습니다:

1. ✅ **물리 기반 AI의 개념**
   - 데이터 + 물리 = 더 강력한 모델

2. ✅ **PDE를 Loss에 통합하는 방법**
   - Automatic differentiation 활용

3. ✅ **일반화 능력의 중요성**
   - 학습 범위 밖 예측 가능

4. ✅ **CAE vs AI vs PINN 비교**
   - 각 방법의 장단점 이해

---

## 🤝 기여

이 프로젝트는 교육 목적으로 제작되었습니다.
개선 사항이나 버그는 Issue를 통해 제보해주세요!

---

## 📄 라이선스

MIT License

---

## 👨‍💻 제작

**Physics-Informed Neural Networks Educational Demo**

*"AI가 물리 법칙을 학습한다"*

---

## 🔥 Quick Start 요약

```bash
# 1. 클론 또는 다운로드
cd Base_PINN

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 실행
streamlit run app.py

# 4. 브라우저에서 http://localhost:8501 접속

# 5. 좌측 설정 → Run Simulation → 결과 확인!
```

**핵심을 기억하세요:**
- CAE: 방정식을 **인간이 푼다**
- Pure AI: 데이터 패턴을 **배운다**
- PINN: **물리 법칙을 학습한다** ← 🎯

---

**Happy Learning! 🚀**
