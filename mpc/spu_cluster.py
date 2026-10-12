"""병원별 프로세스로 SecretFlow + SPU를 띄우는 프로덕션 모드 초기화.

simulate_spu._make_spu()는 sf.init(parties=..., address="local")로 한 프로세스 안에서
3개 병원을 흉내 낸다 (시뮬레이션 모드). 여기서는 병원마다 별도 프로세스를 띄우고,
각 프로세스가 자기 병원(self_party)으로만 동작한다.

병원당 포트 2개 (cluster_local.json):
  sf_port  -- SecretFlow 노드 간 통신. PYU 객체를 다른 병원/SPU로 옮길 때 (gRPC)
  spu_port -- SPU 런타임 링크. SPU 연산 중 ABY3 비밀분산 조각 교환 (brpc)
두 포트는 각기 다른 서버가 bind하므로 같은 머신에서 같은 번호를 쓸 수 없다.

ray_mode=False 프로덕션 모드: 병원마다 Ray 클러스터를 따로 띄우지 않고, 자기 소유
PYU 작업은 자기 프로세스 안에서 바로 실행한다. 중앙 드라이버는 없다 -- 세 프로세스가
같은 스크립트를 실행하고 self_party만 다르다.
"""

import json
import os

import secretflow as sf

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_local.json")


def load_cluster_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def init_party(self_party, config):
    """self_party로 SecretFlow를 초기화하고 (spu, pyus) 반환. pyus 순서는 config 순서."""
    parties = config["parties"]
    assert self_party in parties, f"unknown party {self_party}; choices: {list(parties)}"

    sf_cluster_config = {
        "parties": {
            name: {
                "address": f"{p['host']}:{p['sf_port']}",
                "listen_addr": f"0.0.0.0:{p['sf_port']}",
            }
            for name, p in parties.items()
        },
        "self_party": self_party,
    }
    # 병원 프로세스가 수십 초씩 늦게 떠도(WSL에서 /mnt/d import가 느림) 첫 전송이
    # 기본 60초 deadline에 걸려 죽지 않도록 타임아웃과 재시도 간격을 늘린다.
    comm_options = {
        "timeout_in_ms": 300 * 1000,
        "grpc_retry_policy": {
            "maxAttempts": 5,
            "initialBackoff": "5s",
            "maxBackoff": "60s",
            "backoffMultiplier": 2,
            "retryableStatusCodes": ["UNAVAILABLE"],
        },
    }
    sf.init(address="local", cluster_config=sf_cluster_config, ray_mode=False,
            cross_silo_comm_options=comm_options)

    spu_cluster_def = {
        "nodes": [
            {
                "party": name,
                "address": f"{p['host']}:{p['spu_port']}",
                "listen_address": f"0.0.0.0:{p['spu_port']}",
            }
            for name, p in parties.items()
        ],
        # simulate_spu와 같은 설정 (testing.cluster_def의 3자 기본값: ABY3 + FM128)
        "runtime_config": {"protocol": "ABY3", "field": "FM128"},
    }
    # 다른 병원 프로세스가 늦게 떠도 SPU 링크 연결을 기다리도록 재시도를 넉넉히 준다.
    link_desc = {
        "connect_retry_times": 60,
        "connect_retry_interval_ms": 1000,
        "recv_timeout_ms": 600 * 1000,
    }
    spu = sf.SPU(spu_cluster_def, link_desc=link_desc)
    pyus = [sf.PYU(name) for name in parties]
    return spu, pyus
