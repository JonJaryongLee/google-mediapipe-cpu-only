# MediaPipe 실습

Google MediaPipe 의 Hand Landmarker 와 Selfie Segmentation 두 모델을 CPU 환경에서 실행하고, 단일 이미지·영상 양쪽에 대해 추론 속도(latency, 한 번 처리에 걸리는 시간) 와 그 변동을 측정한 기록이다. GPU 가 없는 환경을 전제로 한다.

## 1. 배경

### 1.1 용어
- **추론(inference)**: 학습이 끝난 모델에 입력을 넣어 결과를 얻는 과정이다. 여기서는 한 장의 사진에서 손 좌표나 사람 마스크를 얻는 일이 추론에 해당한다.
- **Segmentation**: 이미지의 모든 픽셀을 사전 정의된 클래스(배경, 사람, 사물 등)로 분류하는 작업이다.
- **Landmark**: 손가락 마디나 관절처럼 의미 있는 특징점의 좌표 $(x, y, z)$ 를 찾는 작업이다.
- **MediaPipe**: Google 이 공개한 온디바이스(on-device) 추론 라이브러리다. 외부 서버를 거치지 않고 사용자의 PC 에서 모델을 직접 실행한다는 뜻이다. 내부적으로는 모바일·임베디드용 경량 모델 실행기인 TFLite 를 사용하며, CPU 만으로 실시간에 가까운 처리가 가능하다.

### 1.2 다루는 모델

| 모델 | 작업 | 출력 |
|------|------|------|
| Hand Landmarker | 손의 위치를 찾고 21개 관절 좌표 추출 | 점 21개의 $(x, y, z)$ |
| Selfie Segmentation | 사람과 배경을 픽셀 단위로 분리 | 픽셀별 사람 확률(0~1)을 담은 1채널 마스크 |

비유하면 Landmark 는 사진 위에 점을 찍는 작업이고("여기가 검지 끝이다"), Segmentation 은 사진을 색칠하는 작업이다("이 픽셀은 사람, 저 픽셀은 배경").

### 1.3 자주 마주치는 함정

1. **색공간 차이**. 컴퓨터에서 컬러 이미지는 빨강·초록·파랑 세 채널의 순서로 표현되는데, OpenCV 의 `cv2.imread()` 는 결과를 BGR 순서로 돌려주지만 MediaPipe 는 RGB 를 기대한다. 변환을 빼먹으면 모델이 빨강과 파랑을 뒤바꾼 채 추론하여 정확도가 떨어진다.
   ```python
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   ```
2. **모델 재사용**. MediaPipe 솔루션은 그래프(처리 파이프라인)를 메모리에 한 번 올린 뒤 재사용해야 빠르다. 매 프레임마다 새로 인스턴스를 생성하면 모델을 다시 로드하는 초기화 비용이 누적되어 매우 느려진다. 파이썬의 `with` 블록으로 한 번만 열어 두고 그 안에서 반복 호출한다.

## 2. 환경 구성

- Ubuntu 22.04, Python 3.11 기준이다.
- GPU 는 사용하지 않으며, 이하 모든 코드는 CPU 에서 동작한다.

가상환경을 만들고 패키지를 설치한다. 아래 예시는 빠른 패키지 관리자인 `uv` 를 사용하지만, 동일한 효과는 표준 `python -m venv` 와 `pip` 으로도 얻을 수 있다.

```bash
uv venv --python 3.11
. .venv/bin/activate
uv pip install mediapipe==0.10.11 opencv-python==4.13.0.92 ipykernel==7.2.0 ipywidgets==8.1.8 seaborn==0.13.2
```

다음 파이썬 코드를 실행해 모델 가중치를 미리 받아 두자.

```python
import mediapipe as mp

with mp.solutions.hands.Hands(static_image_mode=True):
    pass
with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1):
    pass
```

작업 폴더와 샘플 파일은 다음 구조로 둔다.

```bash
mkdir ./samples/ ./outputs/
```

총 세 가지 파일을 인터넷에서 찾아 다운로드받자.  

| 파일 | 용도 |
|------|------|
| `samples/hands.jpg` | 손이 정면으로 잘 보이는 사진 (단일 이미지 추론용) |
| `samples/person.jpg` | 인물 사진 (단일 이미지 분리용) |
| `samples/hello.mp4` | 손을 흔드는 짧은 영상 (영상 추론용) |

`hello.mp4` 는 Pexels 의 무료 소스를 사용한다.
- https://www.pexels.com/video/a-man-greeting-and-waving-his-hand-4586958/
- https://www.pexels.com/video/a-man-waving-his-hands-8627747/

## 3. Hand Landmarker

### 3.1 단일 이미지 추론

이미지 한 장에서 21개 관절을 검출하고, 점·선이 그려진 결과 이미지를 저장한다. 본문에서 등장하는 `mp_drawing` 은 검출 결과를 이미지 위에 그려 주는 보조 모듈이고, `mp_styles` 는 점과 선의 색·두께 같은 기본 스타일을 제공한다.

```python
import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

img = cv2.imread("./samples/hands.jpg")
if img is None:
    raise FileNotFoundError("./samples/hands.jpg 가 없습니다.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# static_image_mode=True : 한 장씩 독립 처리(추적 비활성화)
# max_num_hands=2        : 한 프레임에서 최대 손 개수
# min_detection_confidence=0.5 : 이 값 미만은 손이 아닌 것으로 무시
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
) as hands:
    t0 = time.perf_counter()
    result = hands.process(img_rgb)
    latency_ms = (time.perf_counter() - t0) * 1000

print(f"Latency: {latency_ms:.2f} ms")

# result.multi_hand_landmarks : 검출된 손들의 리스트(없으면 None)
# 각 hand_landmarks 의 .landmark[0..20] 에 21개 점의 (x, y, z)
if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,                                  # BGR 이미지에 직접 그림
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )
        h, w = img.shape[:2]
        for i, lm in enumerate(hand_landmarks.landmark):
            print(f"#{i:02d}  norm=({lm.x:.3f},{lm.y:.3f},{lm.z:.3f})  "
                  f"pixel=({int(lm.x*w)},{int(lm.y*h)})")
else:
    print("손이 검출되지 않았습니다.")

cv2.imwrite("./outputs/hand_landmarks.jpg", img)
```

좌표 해석은 다음과 같다.

- `lm.x`, `lm.y` 는 이미지 크기에 상관없이 0~1 사이로 환산된 정규화 좌표다. 0 이 좌상단, 1 이 우하단을 의미한다.
- 실제 픽셀 좌표가 필요하면 이미지의 너비·높이를 곱한다. 예를 들어 가로 1920 픽셀 이미지에서 `lm.x = 0.5` 는 픽셀 기준으로 960 에 해당한다.
- `lm.z` 는 손목(0번 점) 을 기준으로 한 상대 깊이이며, 음수면 카메라에 더 가깝다.

### 3.2 영상 추론

연속 프레임에서는 매 프레임마다 손을 처음부터 찾는 대신, 직전 프레임에서 찾은 위치를 이어 따라가는 방식(추적, tracking) 을 활용한다. 검출보다 훨씬 가벼워 영상에서는 이 쪽이 유리하다. `static_image_mode=False` 로 두면 추적이 켜지고, 추적 신뢰도의 하한을 `min_tracking_confidence` 로 지정한다. 결과 영상 저장에는 OpenCV 의 영상 기록기 `cv2.VideoWriter` 가 추가된다.

```python
import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

SRC = "./samples/hello.mp4"
DST = "./outputs/hands_hello.mp4"

cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    raise RuntimeError(f"cannot open {SRC}")

# 입력 영상의 메타데이터를 그대로 출력에 사용
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0     # 일부 컨테이너는 0 을 반환

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(DST, fourcc, src_fps, (w, h))

latencies = []

with mp_hands.Hands(
    static_image_mode=False,                    # 영상이므로 추적 모드
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False             # 내부 복사 회피, 약간의 속도 이득

        t0 = time.perf_counter()
        result = hands.process(rgb)
        latencies.append((time.perf_counter() - t0) * 1000)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        avg_ms = sum(latencies) / len(latencies)
        cv2.putText(frame, f"avg {avg_ms:.1f} ms (~{1000/avg_ms:.1f} FPS)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        writer.write(frame)

cap.release()
writer.release()

print(f"frames: {len(latencies)}  "
      f"avg latency: {sum(latencies)/len(latencies):.2f} ms")
```

자주 막히는 지점은 두 가지다.

- 결과 영상이 0KB 로 만들어지는 경우는 입력 영상의 가로·세로 크기 `(w, h)` 와 `writer` 의 `(w, h)` 가 다를 때다. 위 코드처럼 입력에서 읽은 값을 그대로 사용한다.
- 영상이 열리지 않으면 경로·확장자를 먼저 확인한다. 파일 형식에 따라 `mp4v` 외에 `avc1` 같은 다른 코덱이 필요한 환경도 있다.

### 3.3 21개 랜드마크 인덱스

손목이 0번이고, 엄지(1 부터 4) → 검지(5 부터 8) → 중지(9 부터 12) → 약지(13 부터 16) → 소지(17 부터 20) 순으로 번호가 매겨진다. 각 손가락은 4개 점이며 끝(TIP) 이 가장 큰 번호다.

| Index | 부위              | Index | 부위              |
|------:|-------------------|------:|-------------------|
| 0     | WRIST (손목)      | 11    | MIDDLE_FINGER_DIP |
| 1     | THUMB_CMC         | 12    | MIDDLE_FINGER_TIP |
| 2     | THUMB_MCP         | 13    | RING_FINGER_MCP   |
| 3     | THUMB_IP          | 14    | RING_FINGER_PIP   |
| 4     | THUMB_TIP (엄지 끝) | 15  | RING_FINGER_DIP   |
| 5     | INDEX_FINGER_MCP  | 16    | RING_FINGER_TIP   |
| 6     | INDEX_FINGER_PIP  | 17    | PINKY_MCP         |
| 7     | INDEX_FINGER_DIP  | 18    | PINKY_PIP         |
| 8     | INDEX_FINGER_TIP (검지 끝) | 19 | PINKY_DIP    |
| 9     | MIDDLE_FINGER_MCP | 20    | PINKY_TIP (소지 끝) |
| 10    | MIDDLE_FINGER_PIP |       |                   |

약어의 의미는 다음과 같다.
- MCP: 손바닥쪽 첫 마디(주먹 쥐었을 때 튀어나오는 곳)
- PIP/DIP 는 가운데/끝쪽 마디
- TIP 은 손가락 끝
- 엄지는 구조가 달라 CMC/MCP/IP/TIP 로 부른다.

응용 예로 검지 끝(8) 과 엄지 끝(4) 사이 거리를 임계값과 비교하면 OK 사인 여부를 판정할 수 있다.

```python
import math
distance = math.hypot(landmark[4].x - landmark[8].x,
                      landmark[4].y - landmark[8].y)
```

## 4. Selfie Segmentation

### 4.1 단일 이미지 분리

컬러 인물 사진을 입력하면 같은 해상도의 1채널 확률맵(이하 마스크) 이 결과로 나온다. 마스크는 원본과 가로·세로가 같은 회색조 이미지로, 각 픽셀 값은 0~1 사이 실수이며 그 자리가 사람일 확률을 뜻한다. 0.5 를 기준으로 사람과 배경을 가르고(이진화), 사람 영역은 그대로 두고 배경만 회색으로 치환한다.

입력은 컬러 그대로 사용한다. 모델이 1채널 마스크를 추가로 반환할 뿐이며, 최종 합성 결과는 원본 색을 유지한 컬러 이미지다.

```
컬러 사진 (H, W, 3) ─[모델]→ 마스크 (H, W) ─[합성]→ 배경 치환된 컬러 이미지 (H, W, 3)
```

```python
import cv2
import numpy as np
import mediapipe as mp
import time

mp_selfie = mp.solutions.selfie_segmentation

img = cv2.imread("./samples/person.jpg")
if img is None:
    raise FileNotFoundError("./samples/person.jpg 가 존재하지 않습니다.")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# model_selection=0 : general(256x256). 정확도 우위
# model_selection=1 : landscape(256x144). 속도 우위
with mp_selfie.SelfieSegmentation(model_selection=1) as selfie:
    t0 = time.perf_counter()
    result = selfie.process(rgb)
    latency_ms = (time.perf_counter() - t0) * 1000

mask = result.segmentation_mask                       # (H, W) float, 0(배경) ~ 1(사람)
condition = np.stack((mask,) * 3, axis=-1) > 0.5      # 0.5 초과 픽셀만 True 인 3채널 boolean

bg = np.zeros(img.shape, dtype=np.uint8)
bg[:] = (192, 192, 192)                               # BGR 회색으로 채운 배경 이미지

output = np.where(condition, img, bg)                 # True 인 픽셀은 원본, 나머지는 회색

print(f"Latency: {latency_ms:.2f} ms")
cv2.imwrite("./outputs/selfie_mask.png", (mask * 255).astype(np.uint8))
cv2.imwrite("./outputs/selfie_composite.jpg", output)
```

임계값 0.5 는 합성 시 사람과 배경을 가르는 기준이다. 이 값을 낮추면 사람으로 인정되는 영역이 넓어져 머리카락 같은 모호한 부분도 살아나지만 배경이 섞여 들어온다. 반대로 높이면 확실한 부분만 남아 깔끔하지만 가장자리가 잘려 나간다.

### 4.2 영상에서 배경 약화

`hello.mp4` 에서 사람은 원본 그대로 두고 배경만 약화시켜 `./outputs/selfie_hello.mp4` 로 저장한다. 단일 효과로는 차이가 작아 세 가지를 동시에 적용한다.

1. **강한 블러**: 커널을 키워 형체를 거의 알아볼 수 없게 한다.
2. **흑백 처리**: 배경의 채도를 제거하여 사람만 색이 살아 보이도록 한다.
3. **어둡게**: 픽셀 값을 0.6 배로 낮추어 대비를 추가한다.

핵심 아이디어는 한 프레임에서 두 가지 버전(원본/약화된 배경) 을 만들고 마스크를 기준으로 픽셀별로 골라 합치는 것이다.

```python
import cv2
import numpy as np
import mediapipe as mp
import time

mp_selfie = mp.solutions.selfie_segmentation

SRC = "./samples/hello.mp4"
DST = "./outputs/selfie_hello.mp4"

BLUR_KERNEL = (99, 99)   # 클수록 더 흐림. 홀수만 가능
BG_DARKEN  = 0.6         # 0.0(검정) ~ 1.0(원본)
DESATURATE = True        # True 면 배경을 흑백톤으로

cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    raise RuntimeError(f"cannot open {SRC}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(DST, fourcc, src_fps, (w, h))

latencies = []

with mp_selfie.SelfieSegmentation(model_selection=1) as selfie:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        result = selfie.process(rgb)
        latencies.append((time.perf_counter() - t0) * 1000)

        mask = result.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.5

        # 약화된 배경 만들기
        bg = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
        if DESATURATE:
            gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        bg = np.clip(bg.astype(np.float32) * BG_DARKEN, 0, 255).astype(np.uint8)

        output = np.where(condition, frame, bg)        # 사람=원본, 배경=약화

        avg_ms = sum(latencies) / len(latencies)
        cv2.putText(output, f"avg {avg_ms:.1f} ms (~{1000/avg_ms:.1f} FPS)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        writer.write(output)

cap.release()
writer.release()
```

파라미터별 강도 가이드는 다음과 같다.

| 효과 | 약하게 | 보통 | 강하게 |
|------|--------|------|--------|
| 블러 커널 | (21, 21) | (55, 55) | (99, 99) 이상 |
| `BG_DARKEN` | 0.85 | 0.6 | 0.3 |
| `DESATURATE` | False | True | True + 단색 배경 |

## 5. 성능 측정

영상 루프 안에서 측정한 시간은 영상 파일을 푸는 디코딩이나 결과를 이미지에 그리는 비용까지 섞여 있다. 모델 자체의 추론 시간만 비교하려면 같은 입력 이미지를 반복 호출하여 별도로 측정한다.

### 5.1 측정 스크립트

```python
import cv2
import time
import numpy as np
import mediapipe as mp

def bench(fn, n=200, warmup=20):
    """fn() 을 warmup 회 워밍업 후 n 회 측정해 (mean, std, p95) 를 반환한다."""
    for _ in range(warmup):       # 첫 호출은 모델 초기화로 매우 느려 통계에서 제외
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)   # ms 단위
    arr = np.array(times)
    return arr.mean(), arr.std(), np.percentile(arr, 95)

img = cv2.imread("./samples/hands.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
mean, std, p95 = bench(lambda: hands.process(rgb))
print(f"[Hand]   mean={mean:.2f}ms  std={std:.2f}  p95={p95:.2f}  ~{1000/mean:.1f} FPS")

selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
mean, std, p95 = bench(lambda: selfie.process(rgb))
print(f"[Selfie] mean={mean:.2f}ms  std={std:.2f}  p95={p95:.2f}  ~{1000/mean:.1f} FPS")
```

각 통계의 의미는 다음과 같다.

- `mean`: 평균 추론 시간. 평소 속도를 나타낸다.
- `std`: 표준편차로, 측정값이 평균에서 얼마나 흩어져 있는지를 보여 준다. 작을수록 시간 변동이 적어 안정적이다.
- `p95`: 95번째 백분위수. 측정 100번 중 95번은 이 값 이하로 끝나고, 나머지 5번 정도가 이 값보다 더 오래 걸린다는 뜻이다. 가끔 발생하는 느린 프레임, 즉 최악에 가까운 시나리오를 가늠하는 지표다.

### 5.2 측정 결과

Ubuntu 22.04, Python 3.11, CPU 환경에서 컬러 입력(`hands.jpg`) 으로 200회 측정한 결과다.

| 모델   | mean (ms) | std (ms) | p95 (ms) | FPS    |
|--------|----------:|---------:|---------:|-------:|
| Hand   | 30.73     | 7.33     | 32.07    | ~32.5  |
| Selfie | 3.65      | 0.28     | 4.19     | ~274.0 |

**(1) 실시간성**. 한 프레임을 33.3 ms 안에 처리하면 초당 30장(30 FPS) 이 되어 사람 눈에 부드럽게 보인다. Hand 는 평균 32.5 FPS 로 30 FPS 마감을 간신히 통과하며, p95 가 32.07 ms 라 한계까지 1.2 ms 밖에 남지 않는다. 60 FPS 입력의 16.7 ms 마감은 충족하지 못한다. Selfie 는 274 FPS 로 한 프레임에 3.65 ms 만 사용하므로 나머지 30 ms 가량을 후처리(블러, 합성, 인코딩) 에 할당할 수 있다.

**(2) 두 모델의 속도 차이**. Hand 가 Selfie 보다 약 8.4 배 느리다. 모델 구조가 다르기 때문이다. Hand Landmarker 는 먼저 사진 안에서 손 위치를 사각형으로 잡는 Palm Detection 을 거치고, 그 사각형 안에서 21개 점의 좌표를 추정하는 Hand Landmark Model 을 한 번 더 돌리는 2단계 구조다. 영상에서 추적이 안정적인 프레임에서는 첫 단계가 생략되지만, 단일 이미지나 추적이 끊긴 첫 프레임에서는 두 단계가 모두 수행된다. 반면 Selfie Segmentation 은 256×144 크기 입력을 이미지 인식에 특화된 신경망(CNN, Convolutional Neural Network) 에 한 번 통과시키면 끝이다.

**(3) 안정성(std)**. 비율로 보면 Hand 23.8%, Selfie 7.7% 로 이번에는 Hand 가 더 들쭉날쭉하다. 추론이 무거운 2단계 파이프라인일수록 OS 가 잠깐 다른 작업을 처리하는 등 외부 요인의 영향이 누적되기 쉬운 반면, 짧은 단일 추론인 Selfie 는 절대 변동(0.28 ms) 자체가 0.5 ms 미만이라 매우 안정적이다.

**(4) 꼬리(p95)**. `p95 - mean` 이 Hand 1.34 ms, Selfie 0.54 ms 로 양쪽 모두 평균 근처에 모여 있다. 다만 Hand 는 std(7.33 ms) 가 `p95 - mean` 보다 훨씬 크다. 95% 의 측정은 평균에 바짝 붙어 있는데도 상위 1~2% 가 그보다 훨씬 멀리 튀어 평균만 끌어올리는, 드물게 매우 느린 프레임이 섞여 들어오는 분포라는 뜻이다. 평소에는 33.3 ms 안에 들어오나 다른 프로그램이 CPU 를 같이 쓰는 등 외부 부하가 있을 때는 일부 프레임이 마감을 놓칠 가능성이 남는다.

**(5) 두 모델의 직렬 호출**. 한 루프에서 두 모델을 차례로 돌리면 추론 시간이 더해진다. `30.73 + 3.65 ≈ 34.4 ms > 33.3 ms` 로 30 FPS 마감을 약 1 ms 초과하며, 합산 FPS 는 약 29 FPS 가 된다. 시각적 차이는 거의 느껴지지 않지만, 엄격하게 30 FPS 를 보장하려면 입력 해상도를 낮추거나 두 모델을 별도 스레드로 분리해 동시에 돌리는 방법을 검토한다.

### 5.3 결론

MediaPipe 는 GPU 없이 CPU 만으로도 실시간 처리가 가능하다. 본 측정에서 Hand 32.5 FPS, Selfie 274 FPS 로 단독 사용 시 두 모델 모두 30 FPS 마감을 만족했고, p95 까지 보아도 드롭 프레임이 발생하지 않았다. 다만 Hand 는 평균만으로도 33.3 ms 마감까지 여유가 약 2.6 ms 에 불과하여 외부 부하에 민감하며, 두 모델을 직렬로 합치면 약 34 ms 로 마감을 살짝 넘긴다. 단일 파이프라인에서 두 모델을 함께 쓰려면 입력 해상도 축소나 스레드 분리를 함께 검토한다.
