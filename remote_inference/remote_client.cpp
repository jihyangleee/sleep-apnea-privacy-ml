// 환자 기기(원격) 쪽: 워치 피처를 암호화해 세 병원 서버로 보내고, 돌아온 암호문을 더해 복호화한다.
// 비밀키는 이 프로세스 밖으로 나가지 않는다. 병원에 보내는 것은 회전키와 입력 암호문뿐이다.
//
// 사용법: remote_client <client_model.txt> <hospitals.conf> <A.csv> <B.csv> <C.csv>
//                       [--id <nsrrid>] [--threshold 0.5] [--out remote_predictions.csv]
// 예:     ./remote_client client_model.txt hospitals.conf watch/A.csv watch/B.csv watch/C.csv --id 202502080459
//
// hospitals.conf: 병원마다 한 줄 "<h> <host> <port>" ('#'로 시작하는 줄은 주석)

#include "common.hpp"

#include <csignal>
#include <future>
#include <limits>

struct Endpoint { std::string host; int port = 0; };

std::array<Endpoint, H> loadEndpoints(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open hospitals config: " + path);
    std::array<Endpoint, H> eps{};
    std::array<bool, H> seen{};
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int h; Endpoint ep;
        if (!(ss >> h >> ep.host >> ep.port)) continue;
        if (h < 0 || h >= static_cast<int>(H)) throw std::runtime_error("Bad hospital id in config: " + line);
        eps[h] = ep; seen[h] = true;
    }
    for (std::size_t h = 0; h < H; ++h)
        if (!seen[h]) throw std::runtime_error("hospitals.conf: missing hospital " + std::to_string(h));
    return eps;
}

// 한 번 연결해 명령 하나를 보내고 답 하나를 받는다.
std::string request(const Endpoint& ep, const std::vector<std::string>& blobs) {
    int fd = connectTo(ep.host, ep.port);
    try {
        for (const auto& b : blobs) sendBlob(fd, b);
        auto reply = recvBlob(fd);
        close(fd);
        return reply;
    } catch (...) { close(fd); throw; }
}

void runRemoteClient(const Model& model, const std::array<Endpoint, H>& eps,
                     std::vector<Patient> patients, double threshold, const std::string& outPath) {
    // 1) 키 생성: 비밀키 sk는 이 기기에만 있다.
    const auto keyStart = Clock::now();
    SKGenerator skgen(PRESET);
    auto sk = skgen.genKey();
    sk->to(Device::CPU);
    const u32 logSlots = sk->logDegree() - 1;
    if (logSlots != LOG_SLOTS)
        throw std::runtime_error("Server expects logSlots=15; actual=" + std::to_string(logSlots));

    EnDecoder encoder(PRESET);
    EnDecryptor crypt(PRESET);
    EncKeyGenerator ekgen(PRESET);
    auto encKey = ekgen.genKey(*sk);
    SwKeyGenerator swkgen(PRESET);
    HomEval eval(PRESET);
    auto k1 = swkgen.genRotKey(*sk, 1);
    auto k2 = swkgen.genRotKey(*sk, 2);
    auto k4 = swkgen.genRotKey(*sk, 4);
    auto k8 = swkgen.genRotKey(*sk, 8);
    std::cout << std::fixed << std::setprecision(3) << "key_gen_ms=" << elapsed(keyStart) << "\n";

    // 2) 회전키를 세 병원에 등록 (병원은 이 키로 암호문을 회전만 할 수 있고 복호화는 못 한다).
    const std::vector<std::string> registerMsg{
        "REGISTER", serialize(*k1), serialize(*k2), serialize(*k4), serialize(*k8)};
    std::size_t keyBytes = 0;
    for (std::size_t i = 1; i < registerMsg.size(); ++i) keyBytes += registerMsg[i].size();
    for (std::size_t h = 0; h < H; ++h) {
        const auto start = Clock::now();
        auto reply = request(eps[h], registerMsg);
        if (reply != "REGISTER_OK")
            throw std::runtime_error("Hospital " + std::to_string(h) + " key registration failed: " + reply);
        std::cout << "Hospital " << h << " (" << eps[h].host << ":" << eps[h].port << ") registered"
                  << " | ms=" << elapsed(start) << " | key_bytes=" << keyBytes << "\n";
    }

    // 3) 환자(밤)마다: 표준화 → 암호화 → 세 병원에 동시 전송 → 결과 암호문 3개 합산 → 복호화
    std::ofstream out(outPath);
    if (!out) throw std::runtime_error("Cannot write " + outPath);
    out << "nsrrid,ckks_logit,probability,predicted_label,true_label,round_trip_ms,request_bytes,response_bytes\n";
    out << std::setprecision(15);

    double timeSum = 0;
    for (const auto& patient : patients) {
        const auto roundStart = Clock::now();
        HealthFeatureVector normalized{};
        for (size_t i = 0; i < N; ++i) normalized[i] = (patient.raw[i] - model.mean[i]) / model.scale[i];
        auto inputMsg = healthFeaturesToMessage(normalized, logSlots);
        inputMsg.to(Device::CPU);
        auto pt = IPlaintext::make(); encoder.encode(inputMsg, *pt);
        auto ct = ICiphertext::make(); crypt.encrypt(*pt, *encKey, *ct);
        const std::string ctBytes = serialize(*ct);

        std::array<std::future<std::string>, H> futures;
        for (std::size_t h = 0; h < H; ++h)
            futures[h] = std::async(std::launch::async, [&eps, h, &ctBytes]() {
                return request(eps[h], {"INFER", ctBytes});
            });
        std::vector<decltype(ICiphertext::make())> results;
        std::size_t responseBytes = 0;
        for (std::size_t h = 0; h < H; ++h) {
            auto reply = futures[h].get();
            if (reply == "KEY_NOT_REGISTERED")
                throw std::runtime_error("Hospital " + std::to_string(h) + " lost the rotation keys (restarted?)");
            responseBytes += reply.size();
            results.push_back(deserialize<ICiphertext>(reply));
        }

        auto total = ICiphertext::make();
        eval.add(*results[0], *results[1], *total);
        eval.add(*total, *results[2], *total);
        auto resultPt = IPlaintext::make(); crypt.decrypt(*total, *sk, *resultPt);
        Message resultMsg; encoder.decode(*resultPt, resultMsg); resultMsg.to(Device::CPU);

        const double logit = resultMsg.data()[0].real() + model.bias;
        const double prob = sigmoid(logit);
        const int pred = prob >= threshold ? 1 : 0;
        const double t = elapsed(roundStart);
        timeSum += t;

        out << patient.id << ',' << logit << ',' << prob << ',' << pred << ',' << patient.label << ','
            << t << ',' << ctBytes.size() * H << ',' << responseBytes << '\n';
        std::cout << patient.id << " | P(AHI>=15)=" << std::setprecision(4) << prob
                  << " | " << (pred ? "위험" : "정상") << " | round_trip_ms=" << std::setprecision(1) << t
                  << std::setprecision(3) << "\n";
    }
    std::cout << "\nnights=" << patients.size()
              << " avg_round_trip_ms=" << (patients.empty() ? 0.0 : timeSum / patients.size())
              << "\nresults=" << outPath << "\nREMOTE CKKS INFERENCE SUCCESS\n";
}

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    try {
        if (argc < 6)
            throw std::runtime_error(
                "Usage: remote_client <client_model.txt> <hospitals.conf> <A.csv> <B.csv> <C.csv>"
                " [--id <nsrrid>] [--threshold 0.5] [--out remote_predictions.csv]");
        std::string onlyId, outPath = "remote_predictions.csv";
        double threshold = DEFAULT_THRESHOLD;
        for (int i = 6; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            if (arg == "--id") onlyId = argv[++i];
            else if (arg == "--threshold") threshold = std::stod(argv[++i]);
            else if (arg == "--out") outPath = argv[++i];
            else throw std::runtime_error("Unknown option: " + arg);
        }
        if (!(threshold > 0.0 && threshold < 1.0)) throw std::runtime_error("Threshold must be in (0, 1)");

        const auto model = loadClientModel(argv[1]);
        const auto eps = loadEndpoints(argv[2]);
        auto patients = loadPatients(argv[3], argv[4], argv[5], model);
        if (!onlyId.empty()) {
            patients.erase(std::remove_if(patients.begin(), patients.end(),
                                          [&](const Patient& p) { return p.id != onlyId; }),
                           patients.end());
            if (patients.empty()) throw std::runtime_error("No row with nsrrid " + onlyId);
        }
        std::cout << "Loaded " << patients.size() << " night(s)\n";
        runRemoteClient(model, eps, std::move(patients), threshold, outPath);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
