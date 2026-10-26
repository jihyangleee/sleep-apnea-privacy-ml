#pragma once
// 원격 CKKS 추론 공통 코드 (hospital_server.cpp, remote_client.cpp가 같이 쓴다).
// inference_v5/DistributedInferenceV5.cpp의 모델 로딩·CSV·TCP 코드를 그대로 옮기고,
// 127.0.0.1 고정이던 주소만 호스트/포트를 받도록 바꿨다.

#include "health_features.hpp"
#include "HEaaN2/HEaaN2.hpp"
#include "HEaaN2/Serialization.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace heaan;

constexpr std::size_t N = NUM_HEALTH_FEATURES;
constexpr std::size_t H = 3;
// 병원별 피처 수 (A: SpO2·심박 7, B: 수면 4, C: 인구통계·설문 5). 합이 N = 16 이어야 회전 1·2·4·8 합산 블록에 맞는다.
constexpr std::array<std::size_t, H> HOSPITAL_SIZES{7, 4, 5};
static_assert(HOSPITAL_SIZES[0] + HOSPITAL_SIZES[1] + HOSPITAL_SIZES[2] == N, "hospital sizes must sum to N");
static_assert(N <= 16, "rotation sum (1,2,4,8) covers at most 16 slots");
constexpr PresetParamsId PRESET = PresetParamsId::F16;
constexpr u32 LOG_SLOTS = 15;
constexpr double DEFAULT_THRESHOLD = 0.5;
constexpr int SOCKET_TIMEOUT_SEC = 60;  // 원격 상대가 멈춰도 무한정 기다리지 않게

struct Model {
    std::array<std::string, N> names{};
    std::array<double, N> mean{};
    std::array<double, N> scale{};
    std::array<std::vector<std::size_t>, H> indices;
    std::array<std::vector<double>, H> weights;
    double bias = 0.0;
};

inline void expect(std::istream& in, const std::string& token) {
    std::string found;
    if (!(in >> found) || found != token)
        throw std::runtime_error("Expected '" + token + "', got '" + found + "'");
}

// 환자 기기용: 표준화 값과 bias만 있다 (가중치 없음).
inline Model loadClientModel(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open client model: " + path);
    Model m;
    expect(in,"HEAAN2_CLIENT_V1"); expect(in,"NUM_FEATURES");
    size_t n; if (!(in>>n) || n!=N) throw std::runtime_error("Invalid client feature count");
    expect(in,"FEATURE_NAMES"); for(auto& v:m.names) if(!(in>>v)) throw std::runtime_error("Bad feature name");
    expect(in,"SCALER_MEAN"); for(auto& v:m.mean) if(!(in>>v)) throw std::runtime_error("Bad scaler mean");
    expect(in,"SCALER_SCALE"); for(auto& v:m.scale) if(!(in>>v)||v<=0) throw std::runtime_error("Bad scaler scale");
    expect(in,"BIAS"); if(!(in>>m.bias)) throw std::runtime_error("Bad bias");
    return m;
}

// 병원 h용: 자기 피처 위치와 가중치만 있다.
inline Model loadHospitalModel(const std::string& path, int h) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open hospital model: " + path);
    Model m;
    expect(in,"HEAAN2_HOSPITAL_V1"); expect(in,"HOSPITAL_ID");
    int id; if(!(in>>id)||id!=h) throw std::runtime_error("Hospital ID mismatch");
    expect(in,"INDICES");
    const size_t count=HOSPITAL_SIZES[h];
    for(size_t j=0;j<count;++j) {size_t idx; if(!(in>>idx)||idx>=N) throw std::runtime_error("Bad index"); m.indices[h].push_back(idx);}
    expect(in,"WEIGHTS");
    for(size_t j=0;j<count;++j) {double w; if(!(in>>w)) throw std::runtime_error("Bad weight"); m.weights[h].push_back(w);}
    return m;
}

// ── CSV (워치 A/B/C.csv, nsrrid 기준) ─────────────────────────────────────────

struct Patient { std::string id; HealthFeatureVector raw{}; double ahi=-1; int label=-1; };

inline std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> cells; std::string current; bool quoted=false;
    for(char c:line){ if(c=='"') quoted=!quoted; else if(c==','&&!quoted){cells.push_back(current);current.clear();} else if(c!='\r') current+=c; }
    cells.push_back(current); return cells;
}
struct CsvTable {std::vector<std::string> columns; std::unordered_map<std::string,std::vector<std::string>> rows;};
inline CsvTable readCsv(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("Cannot open CSV: "+path);
    std::string line; if(!std::getline(in,line)) throw std::runtime_error("Empty CSV");
    CsvTable table; table.columns=splitCsv(line);
    while(std::getline(in,line)) {
        if(line.empty()||line=="\r") continue;
        auto fields=splitCsv(line); if(fields.size()!=table.columns.size()) throw std::runtime_error("Invalid CSV column count");
        auto id=fields[0]; if(!table.rows.emplace(id,std::move(fields)).second) throw std::runtime_error("Duplicate nsrrid: "+id);
    }
    return table;
}
inline bool hasColumn(const CsvTable& t, const std::string& col) {
    return std::find(t.columns.begin(),t.columns.end(),col)!=t.columns.end();
}
inline double getNumber(const CsvTable& t, const std::string& id, const std::string& col) {
    auto it=std::find(t.columns.begin(),t.columns.end(),col);
    if(it==t.columns.end()) throw std::runtime_error("Missing CSV feature: "+col);
    auto row=t.rows.find(id); if(row==t.rows.end()) throw std::runtime_error("Missing patient ID: "+id);
    const auto& str=row->second[static_cast<size_t>(it-t.columns.begin())];
    size_t used=0; double v=std::stod(str,&used);
    if(used!=str.size()||!std::isfinite(v)) throw std::runtime_error("Invalid number for "+id+"/"+col);
    return v;
}
// 라벨(ahi_a0h3a)은 있으면 읽고, 없거나 음수(워치 데이터는 -1)면 label = -1로 둔다.
inline std::vector<Patient> loadPatients(const std::string& a,const std::string& b,const std::string& c,const Model& model) {
    auto A=readCsv(a), B=readCsv(b), C=readCsv(c);
    std::vector<std::string> ids; for(const auto& pair:A.rows) ids.push_back(pair.first);
    std::sort(ids.begin(),ids.end()); std::vector<Patient> patients;
    for(const auto& id:ids){
        if(!B.rows.count(id)||!C.rows.count(id)) throw std::runtime_error("Mismatched patient IDs: "+id);
        Patient p; p.id=id;
        for(size_t i=0;i<N;++i) {
            const auto& name=model.names[i];
            const CsvTable* t=nullptr;
            if(hasColumn(A,name)) t=&A;
            if(hasColumn(B,name)) {if(t) throw std::runtime_error("Duplicate feature across CSVs");t=&B;}
            if(hasColumn(C,name)) {if(t) throw std::runtime_error("Duplicate feature across CSVs");t=&C;}
            if(!t) throw std::runtime_error("Missing feature: "+name);
            p.raw[i]=getNumber(*t,id,name);
        }
        if(hasColumn(B,"ahi_a0h3a")) {
            p.ahi=getNumber(B,id,"ahi_a0h3a");
            p.label=p.ahi<0?-1:(p.ahi>=15.0?1:0);
        }
        patients.push_back(p);
    }
    return patients;
}

inline Message maskMessage(const Model& model, std::size_t hospital, u32 logSlots) {
    Message msg(logSlots);
    const std::size_t slots = std::size_t{1} << logSlots;
    auto* p = msg.data();
    for (std::size_t i = 0; i < slots; ++i) p[i] = Complex(0.0, 0.0);
    for (std::size_t j = 0; j < model.indices[hospital].size(); ++j)
        p[model.indices[hospital][j]] = Complex(model.weights[hospital][j], 0.0);
    return msg;
}

inline double sigmoid(double x) {
    return x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

// ── 시간 측정 ─────────────────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// ── TCP: 길이 접두 바이너리 메시지 ────────────────────────────────────────────

inline void sendAll(int fd, const void* ptr, size_t n) {
    const char* p = static_cast<const char*>(ptr);
    while (n) {
        ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
        if (r <= 0) throw std::runtime_error("TCP send failed");
        p += r;
        n -= static_cast<size_t>(r);
    }
}

inline void recvAll(int fd, void* ptr, size_t n) {
    char* p = static_cast<char*>(ptr);
    while (n) {
        ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) throw std::runtime_error("TCP receive failed (timeout or closed)");
        p += r;
        n -= static_cast<size_t>(r);
    }
}

// Length-prefixed binary payload; portable 32-bit network-order halves.
inline void sendBlob(int fd, const std::string& bytes) {
    uint64_t n = bytes.size();
    uint32_t words[2] = {
        htonl(static_cast<uint32_t>(n >> 32)),
        htonl(static_cast<uint32_t>(n))
    };
    sendAll(fd, words, sizeof(words));
    if (n) sendAll(fd, bytes.data(), n);
}

inline std::string recvBlob(int fd) {
    uint32_t words[2];
    recvAll(fd, words, sizeof(words));
    uint64_t n = (uint64_t(ntohl(words[0])) << 32) | ntohl(words[1]);
    if (n > 150000000ULL)
        throw std::runtime_error("Payload exceeds 150 MB limit");
    std::string bytes(static_cast<size_t>(n), '\0');
    if (n) recvAll(fd, bytes.data(), bytes.size());
    return bytes;
}

template<class T>
std::string serialize(const T& object) {
    std::ostringstream out(std::ios::out | std::ios::binary);
    serial::save(out, object);
    return out.str();
}

template<class T>
auto deserialize(const std::string& bytes) {
    std::istringstream in(bytes, std::ios::in | std::ios::binary);
    return serial::loadAsPtr<T>(in, Device::CPU);
}

inline void setTimeouts(int fd) {
    timeval tv{};
    tv.tv_sec = SOCKET_TIMEOUT_SEC;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

// host는 IP 또는 도메인 이름.
inline int connectTo(const std::string& host, int port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res)
        throw std::runtime_error("Cannot resolve host " + host);
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); throw std::runtime_error("socket failed"); }
    setTimeouts(fd);
    const int rc = connect(fd, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);
    if (rc < 0) {
        close(fd);
        throw std::runtime_error("connect failed: " + host + ":" + std::to_string(port));
    }
    return fd;
}

// bindIp가 "0.0.0.0"이면 모든 네트워크 인터페이스에서 받는다 (원격 접속용).
inline int listenOn(const std::string& bindIp, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, bindIp.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        throw std::runtime_error("Invalid bind IP: " + bindIp);
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        listen(fd, 8) != 0) {
        close(fd);
        throw std::runtime_error("bind/listen failed on " + bindIp + ":" + std::to_string(port));
    }
    return fd;
}
