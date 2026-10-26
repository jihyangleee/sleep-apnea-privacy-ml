// 병원 h의 원격 추론 서버: 암호문을 받아 자기 가중치로 부분 logit을 계산하고, 암호문 그대로 돌려준다.
// 비밀키는 받지 않는다. 받는 것은 회전키 4개(REGISTER)와 입력 암호문(INFER)뿐이다.
//
// 사용법: hospital_server <h> <hospital_model.txt> [bind_ip=0.0.0.0] [port=18080+h]
// 예:     ./hospital_server 0 hospital0_model.txt 0.0.0.0 18080
//
// 프로토콜 (모든 메시지는 8바이트 길이 + 본문):
//   REGISTER, rotkey1, rotkey2, rotkey4, rotkey8  ->  REGISTER_OK
//   INFER, ciphertext                             ->  ciphertext(enc(w_h · x_h)) | KEY_NOT_REGISTERED
//   STOP                                          ->  STOP_OK (서버 종료)
// 회전키는 마지막으로 등록한 클라이언트 것 하나만 보관한다 (실습용: 동시에 환자 1명).

#include "common.hpp"

#include <csignal>

void runHospital(int h, const Model& model, const std::string& bindIp, int port) {
    int listener = listenOn(bindIp, port);
    std::cout << "Hospital " << h << " listening on " << bindIp << ":" << port << std::endl;

    EnDecoder encoder(PRESET);
    HomEval eval(PRESET);

    // 가중치 마스크는 요청마다 같으므로 한 번만 인코딩한다.
    auto maskMsg = maskMessage(model, h, LOG_SLOTS);
    maskMsg.to(Device::CPU);
    auto maskPt = IPlaintext::make();
    encoder.encode(maskMsg, *maskPt);

    decltype(serial::loadAsPtr<ISwKey>(
        std::declval<std::istream&>(), Device::CPU)) k1, k2, k4, k8;
    bool keysRegistered = false;

    while (true) {
        sockaddr_in peer{};
        socklen_t peerLen = sizeof(peer);
        int fd = accept(listener, reinterpret_cast<sockaddr*>(&peer), &peerLen);
        if (fd < 0) {
            std::cerr << "accept failed\n";
            continue;
        }
        setTimeouts(fd);
        char peerIp[INET_ADDRSTRLEN] = "?";
        inet_ntop(AF_INET, &peer.sin_addr, peerIp, sizeof(peerIp));

        bool stop = false;
        try {
            const auto requestStart = Clock::now();
            std::string command = recvBlob(fd);

            if (command == "REGISTER") {
                auto b1 = recvBlob(fd);
                auto b2 = recvBlob(fd);
                auto b4 = recvBlob(fd);
                auto b8 = recvBlob(fd);
                k1 = deserialize<ISwKey>(b1);
                k2 = deserialize<ISwKey>(b2);
                k4 = deserialize<ISwKey>(b4);
                k8 = deserialize<ISwKey>(b8);
                keysRegistered = true;
                sendBlob(fd, "REGISTER_OK");
                std::cout << "Hospital " << h << " KEY REGISTERED from " << peerIp
                          << " | time_ms=" << elapsed(requestStart)
                          << " | key_bytes=" << b1.size()+b2.size()+b4.size()+b8.size()
                          << std::endl;
            }
            else if (command == "INFER") {
                // 암호문은 먼저 끝까지 받는다 -- 안 받고 답하면 클라이언트 전송이 막힌다.
                auto ctBytes = recvBlob(fd);
                if (!keysRegistered) {
                    sendBlob(fd, "KEY_NOT_REGISTERED");
                } else {
                    double receiveMs = elapsed(requestStart);
                    const auto evalStart = Clock::now();
                    auto inputCt = deserialize<ICiphertext>(ctBytes);

                    // enc(x) ⊙ mask_h → 슬롯 0에 w_h · x_h가 모이도록 회전 1·2·4·8로 합산
                    auto weighted = ICiphertext::make();
                    eval.mul(*inputCt, *maskPt, *weighted);
                    eval.rescale(*weighted, *weighted);
                    const std::array<int,4> shifts{1,2,4,8};
                    for (int i = 0; i < 4; ++i) {
                        auto rotated = ICiphertext::make();
                        if (i == 0) eval.rot(*weighted, shifts[i], *rotated, *k1);
                        if (i == 1) eval.rot(*weighted, shifts[i], *rotated, *k2);
                        if (i == 2) eval.rot(*weighted, shifts[i], *rotated, *k4);
                        if (i == 3) eval.rot(*weighted, shifts[i], *rotated, *k8);
                        eval.add(*weighted, *rotated, *weighted);
                    }
                    double evalMs = elapsed(evalStart);

                    auto outputBytes = serialize(*weighted);
                    sendBlob(fd, outputBytes);
                    std::cout << std::fixed << std::setprecision(3)
                              << "Hospital " << h << " INFERENCE from " << peerIp
                              << " | receive_ms=" << receiveMs
                              << " | he_eval_ms=" << evalMs
                              << " | request_bytes=" << ctBytes.size()
                              << " | response_bytes=" << outputBytes.size()
                              << std::endl;
                }
            }
            else if (command == "STOP") {
                sendBlob(fd, "STOP_OK");
                stop = true;
            }
            else {
                sendBlob(fd, "UNKNOWN_COMMAND");
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Hospital " << h << " ERROR (" << peerIp << "): " << e.what() << std::endl;
        }
        close(fd);
        if (stop) break;
    }
    close(listener);
}

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    try {
        if (argc < 3)
            throw std::runtime_error("Usage: hospital_server <h> <hospital_model.txt> [bind_ip=0.0.0.0] [port=18080+h]");
        const int h = std::stoi(argv[1]);
        if (h < 0 || h >= static_cast<int>(H)) throw std::runtime_error("Invalid hospital ID");
        const std::string bindIp = argc >= 4 ? argv[3] : "0.0.0.0";
        const int port = argc >= 5 ? std::stoi(argv[4]) : 18080 + h;
        runHospital(h, loadHospitalModel(argv[2], h), bindIp, port);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
