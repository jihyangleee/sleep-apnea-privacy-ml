# 원격 CKKS 추론 실습

환자 기기(원격)가 워치 피처를 암호화해 세 병원 서버로 보내고, 병원은 암호문 상태로 부분 logit을 계산해 돌려준다. 복호화는 환자 기기에서만 한다.

```
[원격 환자 기기] remote_client          [병원 h 서버] hospital_server h
  키 생성 (sk는 밖으로 안 나감)
  REGISTER + 회전키 1·2·4·8   ───────▶  회전키 보관
  표준화 → 암호화
  INFER + 암호문              ───────▶  enc(x) ⊙ 가중치 마스크 → 회전 합산
  세 결과 암호문 합산 → 복호화 ◀───────  enc(w_h · x_h)
  + bias → σ → 위험 판정
```

| 파일 | 내용 |
|---|---|
| `common.hpp` | 모델 로딩, CSV, TCP(길이 접두 메시지), 직렬화. `inference_v5/DistributedInferenceV5.cpp`에서 옮김 |
| `hospital_server.cpp` | 병원 서버. `0.0.0.0`에서 받는다 |
| `remote_client.cpp` | 원격 클라이언트. `hospitals.conf`의 주소로 보낸다 |
| `hospitals.conf` | 병원별 `<h> <host> <port>` |
| `*_model.txt`, `health_features.hpp` | `inference_v5/`에서 복사 (16개 피처 모델) |
| `watch/A,B,C.csv` | 워치 피처 (`inference_v5/watch/`에서 복사) |

## 빌드

`DistributedInferenceV5.cpp`와 같은 방식으로 HEaaN2 devkit의 examples 폴더 아래에서 빌드한다. `health_features.hpp`가 `../examples.hpp`를 include하므로 이 폴더를 `heaan2-examples/` 바로 아래에 둔다. 실행 파일은 두 개다: `hospital_server`, `remote_client`.

원격 기기에서도 클라이언트를 빌드하려면 그 기기에 HEaaN2가 있어야 한다.

## 실행

1. 같은 PC에서 먼저 확인 (`hospitals.conf` 기본값 127.0.0.1):
   ```bash
   ./hospital_server 0 hospital0_model.txt &
   ./hospital_server 1 hospital1_model.txt &
   ./hospital_server 2 hospital2_model.txt &
   ./remote_client client_model.txt hospitals.conf watch/A.csv watch/B.csv watch/C.csv --id 202502080459
   ```
2. 원격으로 실행: 병원 서버 PC의 IP를 `hospitals.conf`에 적고, 클라이언트를 다른 기기에서 실행한다.
   - 서버 방화벽에서 TCP 18080~18082를 연다.
   - 서버가 WSL/Docker 안이면 바깥 포트를 안쪽으로 연결해야 한다 (WSL: `netsh interface portproxy` 또는 mirrored 네트워킹, Docker: `-p 18080:18080`).
   - 같은 공유기 밖에서 접속하면 공유기 포트 포워딩도 필요하다.

결과는 `remote_predictions.csv`(logit, 확률, 판정, 왕복 시간, 송수신 바이트)에 저장된다.

## 실습 범위 밖 (실제 배포 전 필요)

- TLS·서버 인증 없음: 암호문 내용은 안전하지만, 가짜 서버가 엉뚱한 결과를 돌려주는 것은 막지 못한다.
- 서버는 마지막에 등록한 클라이언트의 회전키 하나만 보관한다 (동시에 환자 1명).
- 환자가 병원별 결과 암호문을 각각 복호화할 수 있어, 반복 질의로 가중치를 추정할 수 있다.
