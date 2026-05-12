# 🏆 Matrix-Zero AI Project

본 프로젝트는 $3 \times 3$ 행렬 게임의 최강 전략을 학습하기 위한 AI 엔진과 웹 기반 전략 헬퍼 툴을 포함합니다. GPU(CuPy)를 활용하여 수만 판의 자가대국(Self-play)을 통해 강화학습을 진행합니다.

---

## 🛠 1. 환경 준비 (Prerequisites)

- **OS**: Windows 10/11 (PowerShell 권장)
- **GPU**: NVIDIA GeForce RTX 4050 (또는 CUDA 지원 GPU)
- **Python**: 3.10 버전 이상
- **CUDA Toolkit**: 시스템에 NVIDIA 드라이버 및 CUDA Toolkit이 설치되어 있어야 합니다.

---

## 🚀 2. 빠른 시작 가이드 (GPU 환경 설정)

파워쉘(PowerShell)을 열고 프로젝트 폴더로 이동한 뒤 아래 명령어를 순서대로 입력하세요.

### ① 가상환경 생성 및 활성화
```powershell
# 프로젝트 폴더로 이동
cd "c:\Users\000\OneDrive\Desktop\matrix_game_v2"

# 윈도우용 가상환경 생성
python -m venv venv_win

# 스크립트 실행 권한 허용 (최초 1회만 필요)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 가상환경 활성화
.\venv_win\Scripts\Activate.ps1
```

### ② 라이브러리 설치
`nvidia-smi` 명령어로 확인된 CUDA 버전에 맞춰 CuPy를 설치합니다. (RTX 40 시리즈는 `cuda12x` 권장)
```powershell
# CUDA 버전 확인
nvidia-smi

# CuPy 설치 (CUDA 12.x 이상)
pip install cupy-cuda12x

# 필수 라이브러리 설치
pip install numpy tqdm
```

### ③ GPU 작동 확인
```powershell
python -c "import cupy; print('GPU 활성화 성공:', cupy.is_available())"
```

---

## 🧠 3. AI 학습 및 실행 (Training)

학습을 진행하면 `strategy_weights.json` 파일에 AI의 두뇌(가중치)가 저장됩니다.

```powershell
python main.py
```
1. **Option 2 (Training)** 선택: 30,000 에피소드 자가대국 학습 시작.
2. 학습 완료 후 생성된 `strategy_weights.json`을 확인합니다.

---

## 🌐 4. 웹 전략 헬퍼 사용법

1. 학습된 `strategy_weights.json` 파일을 `index.html`과 같은 폴더에 둡니다.
2. `index.html`을 브라우저로 엽니다.
3. **게임 조언 모드** 혹은 **AI 대국 모드**를 선택하여 실시간 행렬 분석을 활용합니다.

---

## ⚠️ 주의사항

- **GPU 메모리**: 학습 중 `Out of Memory` 에러가 발생하면 `main.py`의 에피소드 당 메모리 정리 주기를 조절하세요.
- **가중치 로드**: 웹 툴에서 가중치가 불러와지지 않는다면 브라우저의 보안 정책(CORS) 때문일 수 있으니, VS Code의 `Live Server` 확장을 사용하거나 깃허브 페이지에 업로드하여 확인하세요.