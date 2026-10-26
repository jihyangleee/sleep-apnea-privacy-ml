#include <unordered_map>
#include "health_features.hpp"
#include "HEaaN2/HEaaN2.hpp"
#include "HEaaN2/Serialization.hpp"
#include "HEaaN2/Serialization.hpp"
#include <filesystem>
#include <sstream>

#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace heaan;

namespace {
constexpr std::size_t N = NUM_HEALTH_FEATURES;
constexpr std::size_t H = 3;
constexpr std::size_t SUM_SLOT = 0; // 16-slot block, right rotations
// 병원별 피처 수 (A: SpO2·심박 7, B: 수면 4, C: 인구통계·설문 5). 합이 N = 16 이어야 회전 1·2·4·8 합산 블록에 맞는다.
constexpr std::array<std::size_t, H> HOSPITAL_SIZES{7, 4, 5};
static_assert(HOSPITAL_SIZES[0] + HOSPITAL_SIZES[1] + HOSPITAL_SIZES[2] == N, "hospital sizes must sum to N");
static_assert(N <= 16, "rotation sum (1,2,4,8) covers at most 16 slots");
// 학습(train_spu_party.py) 정확도와 같은 기준: logit > 0 ⇔ σ > 0.5. 다른 값은 `client <threshold>`로 준다.
constexpr double DEFAULT_THRESHOLD = 0.5;

struct Model {
    std::array<std::string, N> names{};
    std::array<double, N> mean{};
    std::array<double, N> scale{};
    std::array<std::vector<std::size_t>, H> indices;
    std::array<std::vector<double>, H> weights;
    double bias = 0.0;
};

void expect(std::istream& in, const std::string& token) {
    std::string found;
    if (!(in >> found) || found != token)
        throw std::runtime_error("Expected '" + token + "', got '" + found + "'");
}

Model loadModel(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open model: " + filename);
    expect(in, "HEAAN2_LINEAR_V1");
    expect(in, "NUM_FEATURES");
    std::size_t count = 0;
    if (!(in >> count) || count != N) throw std::runtime_error("Expected " + std::to_string(N) + " features");
    Model m;
    expect(in, "FEATURE_NAMES");
    for (auto& v : m.names) if (!(in >> v)) throw std::runtime_error("Bad feature names");
    expect(in, "SCALER_MEAN");
    for (auto& v : m.mean) if (!(in >> v)) throw std::runtime_error("Bad scaler mean");
    expect(in, "SCALER_SCALE");
    for (auto& v : m.scale) if (!(in >> v) || v <= 0) throw std::runtime_error("Bad scaler scale");
    constexpr auto sizes = HOSPITAL_SIZES;
    std::array<bool, N> covered{};
    for (std::size_t h = 0; h < H; ++h) {
        expect(in, "HOSPITAL_" + std::to_string(h) + "_INDICES");
        for (std::size_t j = 0; j < sizes[h]; ++j) {
            std::size_t idx;
            if (!(in >> idx) || idx >= N || covered[idx]) throw std::runtime_error("Bad hospital indices");
            covered[idx] = true;
            m.indices[h].push_back(idx);
        }
        expect(in, "HOSPITAL_" + std::to_string(h) + "_WEIGHTS");
        for (std::size_t j = 0; j < sizes[h]; ++j) {
            double weight;
            if (!(in >> weight)) throw std::runtime_error("Bad hospital weights");
            m.weights[h].push_back(weight);
        }
    }
    if (!std::all_of(covered.begin(), covered.end(), [](bool x) { return x; }))
        throw std::runtime_error("Missing feature index");
    expect(in, "BIAS");
    if (!(in >> m.bias)) throw std::runtime_error("Bad bias");
    return m;
}


Model loadClientModel(const std::string& path) {
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
Model loadHospitalModel(const std::string& path, int h) {
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
struct Patient { std::string id; HealthFeatureVector raw{}; double ahi=0; int label=0; };
std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> cells; std::string current; bool quoted=false;
    for(char c:line){ if(c=='"') quoted=!quoted; else if(c==','&&!quoted){cells.push_back(current);current.clear();} else current+=c; }
    cells.push_back(current); return cells;
}
struct CsvTable {std::vector<std::string> columns; std::unordered_map<std::string,std::vector<std::string>> rows;};
CsvTable readCsv(const std::string& path) {
    std::ifstream in(path); if(!in) throw std::runtime_error("Cannot open CSV: "+path);
    std::string line; if(!std::getline(in,line)) throw std::runtime_error("Empty CSV");
    CsvTable table; table.columns=splitCsv(line);
    while(std::getline(in,line)) {
        if(line.empty()) continue;
        auto fields=splitCsv(line); if(fields.size()!=table.columns.size()) throw std::runtime_error("Invalid CSV column count");
        auto id=fields[0]; if(!table.rows.emplace(id,std::move(fields)).second) throw std::runtime_error("Duplicate nsrrid: "+id);
    }
    return table;
}
double getNumber(const CsvTable& t, const std::string& id, const std::string& col) {
    auto it=std::find(t.columns.begin(),t.columns.end(),col);
    if(it==t.columns.end()) throw std::runtime_error("Missing CSV feature: "+col);
    auto row=t.rows.find(id); if(row==t.rows.end()) throw std::runtime_error("Missing patient ID: "+id);
    const auto& str=row->second[static_cast<size_t>(it-t.columns.begin())];
    size_t used=0; double v=std::stod(str,&used);
    if(used!=str.size()||!std::isfinite(v)) throw std::runtime_error("Invalid number for "+id+"/"+col);
    return v;
}
std::vector<Patient> loadPatients(const std::string& a,const std::string& b,const std::string& c,const Model& model) {
    auto A=readCsv(a), B=readCsv(b), C=readCsv(c);
    if(A.rows.size()!=B.rows.size()||A.rows.size()!=C.rows.size()) throw std::runtime_error("CSV patient counts differ");
    std::vector<std::string> ids; for(const auto& pair:A.rows) ids.push_back(pair.first);
    std::sort(ids.begin(),ids.end()); std::vector<Patient> patients;
    for(const auto& id:ids){
        if(!B.rows.count(id)||!C.rows.count(id)) throw std::runtime_error("Mismatched patient IDs");
        Patient p; p.id=id;
        for(size_t i=0;i<N;++i) {
            const auto& name=model.names[i];
            const CsvTable* t=nullptr;
            if(std::find(A.columns.begin(),A.columns.end(),name)!=A.columns.end()) t=&A;
            if(std::find(B.columns.begin(),B.columns.end(),name)!=B.columns.end()) {if(t) throw std::runtime_error("Duplicate feature across CSVs");t=&B;}
            if(std::find(C.columns.begin(),C.columns.end(),name)!=C.columns.end()) {if(t) throw std::runtime_error("Duplicate feature across CSVs");t=&C;}
            if(!t) throw std::runtime_error("Missing feature: "+name);
            p.raw[i]=getNumber(*t,id,name);
        }
        p.ahi=getNumber(B,id,"ahi_a0h3a"); p.label=p.ahi>=15.0?1:0;
        patients.push_back(p);
    }
    return patients;
}
Message maskMessage(const Model& model, std::size_t hospital, u32 logSlots) {
    Message msg(logSlots);
    const std::size_t slots = std::size_t{1} << logSlots;
    auto* p = msg.data();
    for (std::size_t i = 0; i < slots; ++i) p[i] = Complex(0.0, 0.0);
    for (std::size_t j = 0; j < model.indices[hospital].size(); ++j)
        p[model.indices[hospital][j]] = Complex(model.weights[hospital][j], 0.0);
    return msg;
}

double sigmoid(double x) {
    return x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

} // namespace

// BENCHMARK_TIMING_V1
using Clock = std::chrono::steady_clock;
auto ms = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
};


#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <numeric>
#include <future>
#include <unordered_map>
#include <limits>

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

void sendAll(int fd, const void* ptr, size_t n) {
    const char* p = static_cast<const char*>(ptr);
    while (n) {
        ssize_t r = send(fd, p, n, MSG_NOSIGNAL);
        if (r <= 0) throw std::runtime_error("TCP send failed");
        p += r;
        n -= static_cast<size_t>(r);
    }
}

void recvAll(int fd, void* ptr, size_t n) {
    char* p = static_cast<char*>(ptr);
    while (n) {
        ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) throw std::runtime_error("TCP receive failed");
        p += r;
        n -= static_cast<size_t>(r);
    }
}

// Length-prefixed binary payload; portable 32-bit network-order halves.
void sendBlob(int fd, const std::string& bytes) {
    uint64_t n = bytes.size();
    uint32_t words[2] = {
        htonl(static_cast<uint32_t>(n >> 32)),
        htonl(static_cast<uint32_t>(n))
    };
    sendAll(fd, words, sizeof(words));
    if (n) sendAll(fd, bytes.data(), n);
}

std::string recvBlob(int fd) {
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

int connectTo(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd);
        throw std::runtime_error("connect failed on port " + std::to_string(port));
    }
    return fd;
}

int listenOn(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        listen(fd, 1) != 0) {
        close(fd);
        throw std::runtime_error("bind/listen failed on port " + std::to_string(port));
    }
    return fd;
}


void runHospital(int h, const Model& model) {
    const int port = 18080 + h;
    int listener = listenOn(port);

    std::cout << "Hospital " << h
              << " listening on " << port << std::endl;

    const auto preset = PresetParamsId::F16;
    EnDecoder encoder(preset);
    HomEval eval(preset);

    decltype(serial::loadAsPtr<ISwKey>(
        std::declval<std::istream&>(), Device::CPU)) k1, k2, k4, k8;

    bool keysRegistered = false;

    while (true) {
        int fd = accept(listener, nullptr, nullptr);
        if (fd < 0) {
            std::cerr << "accept failed\n";
            continue;
        }

        try {
            const auto requestStart = Clock::now();

            std::string command = recvBlob(fd);

            if (command == "REGISTER") {
                const auto start = Clock::now();

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

                std::cout
                    << "Hospital " << h
                    << " KEY REGISTERED"
                    << " | time_ms=" << elapsed(start)
                    << " | key_bytes="
                    << b1.size()+b2.size()+b4.size()+b8.size()
                    << std::endl;
            }
            else if (command == "INFER") {
                if (!keysRegistered) {
                    sendBlob(fd, "KEY_NOT_REGISTERED");
                    close(fd);
                    continue;
                }

                auto ctBytes = recvBlob(fd);
                double receiveMs = elapsed(requestStart);

                const auto loadStart = Clock::now();
                auto inputCt = deserialize<ICiphertext>(ctBytes);
                double loadMs = elapsed(loadStart);

                const auto evalStart = Clock::now();

                const u32 logSlots = 15;
                auto maskMsg = maskMessage(model, h, logSlots);
                maskMsg.to(Device::CPU);

                auto maskPt = IPlaintext::make();
                encoder.encode(maskMsg, *maskPt);

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

                const auto saveStart = Clock::now();
                auto outputBytes = serialize(*weighted);
                double saveMs = elapsed(saveStart);

                sendBlob(fd, outputBytes);

                std::cout << std::fixed << std::setprecision(3)
                          << "Hospital " << h << " INFERENCE\n"
                          << "  receive_ms=" << receiveMs << "\n"
                          << "  deserialize_ms=" << loadMs << "\n"
                          << "  he_eval_ms=" << evalMs << "\n"
                          << "  serialize_ms=" << saveMs << "\n"
                          << "  request_bytes=" << ctBytes.size() << "\n"
                          << "  response_bytes=" << outputBytes.size()
                          << std::endl;
            }
            else if (command == "STOP") {
                sendBlob(fd, "STOP_OK");
                close(fd);
                break;
            }
            else {
                sendBlob(fd, "UNKNOWN_COMMAND");
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Hospital " << h
                      << " ERROR: " << e.what() << std::endl;
        }

        close(fd);
    }

    close(listener);
}

void runClient(const Model& model, double threshold) {
    const auto totalStart = Clock::now();

    const std::string root="/home/user/devkit/examples/heaan2-examples/my_example/";
    auto patients=loadPatients(root+"A.csv",root+"B.csv",root+"C.csv",model);
    if(patients.empty()) throw std::runtime_error("No patients");
    std::cout << "Loaded " << patients.size() << " independent test patients\n";

    const auto keyStart = Clock::now();

    const auto preset = PresetParamsId::F16;
    SKGenerator skgen(preset);
    auto sk = skgen.genKey();
    sk->to(Device::CPU);

    const u32 logSlots = sk->logDegree() - 1;
    if (logSlots != 15)
        throw std::runtime_error(
            "Server expects logSlots=15; actual=" +
            std::to_string(logSlots));

    EnDecoder encoder(preset);
    EnDecryptor crypt(preset);
    EncKeyGenerator ekgen(preset);
    auto encKey = ekgen.genKey(*sk);

    SwKeyGenerator swkgen(preset);
    HomEval eval(preset);

    auto k1 = swkgen.genRotKey(*sk, 1);
    auto k2 = swkgen.genRotKey(*sk, 2);
    auto k4 = swkgen.genRotKey(*sk, 4);
    auto k8 = swkgen.genRotKey(*sk, 8);

    double keyMs = elapsed(keyStart);

    const auto keySerializeStart = Clock::now();

    std::array<std::string,4> keyBytes{
        serialize(*k1),
        serialize(*k2),
        serialize(*k4),
        serialize(*k8)
    };

    double keySerializeMs = elapsed(keySerializeStart);

    size_t totalKeyBytes = 0;
    for (const auto& b : keyBytes)
        totalKeyBytes += b.size();

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\n========== KEY REGISTRATION ==========\n";

    double registrationMs[3]{};

    for (int h = 0; h < 3; ++h) {
        const auto start = Clock::now();
        int fd = connectTo(18080 + h);

        sendBlob(fd, "REGISTER");
        for (const auto& b : keyBytes)
            sendBlob(fd, b);

        auto reply = recvBlob(fd);
        close(fd);

        if (reply != "REGISTER_OK")
            throw std::runtime_error(
                "Hospital " + std::to_string(h) +
                " key registration failed: " + reply);

        registrationMs[h] = elapsed(start);

        std::cout << "Hospital " << h
                  << " registration_ms=" << registrationMs[h]
                  << " key_bytes=" << totalKeyBytes
                  << "\n";
    }

    std::ofstream output(root+"v5_predictions.csv");
    if(!output) throw std::runtime_error("Cannot write V5 results");
    output << "nsrrid,ahi_a0h3a,true_label,ckks_logit,ckks_probability,predicted_label,inference_ms\n";
    output << std::setprecision(15);
    std::vector<std::pair<double,int>> scores;
    long long tp=0,tn=0,fp=0,fn=0;
    double timeSum=0, timeSq=0;
    size_t patientCount=0;
    for(const auto& patient:patients) {
        const auto roundStart=Clock::now();
        HealthFeatureVector normalized{};
        for(size_t i=0;i<N;++i) normalized[i]=(patient.raw[i]-model.mean[i])/model.scale[i];
        auto inputMsg=healthFeaturesToMessage(normalized,logSlots);
        inputMsg.to(Device::CPU);
        auto pt=IPlaintext::make(); encoder.encode(inputMsg,*pt);
        auto ct=ICiphertext::make(); crypt.encrypt(*pt,*encKey,*ct);
        const std::string ctBytes=serialize(*ct);
        std::array<std::future<std::string>,3> futures;
        for(int h=0;h<3;++h) {
            futures[h]=std::async(std::launch::async,[h,&ctBytes](){
                int fd=connectTo(18080+h);
                try {
                    sendBlob(fd,"INFER");sendBlob(fd,ctBytes);
                    auto reply=recvBlob(fd);close(fd);
                    if(reply=="KEY_NOT_REGISTERED") throw std::runtime_error("Hospital key cache missing");
                    return reply;
                } catch(...){close(fd);throw;}
            });
        }
        std::vector<decltype(ICiphertext::make())> results;
        for(auto& f:futures) results.push_back(deserialize<ICiphertext>(f.get()));
        auto total=ICiphertext::make();
        eval.add(*results[0],*results[1],*total);
        eval.add(*total,*results[2],*total);
        auto resultPt=IPlaintext::make(); crypt.decrypt(*total,*sk,*resultPt);
        Message resultMsg;encoder.decode(*resultPt,resultMsg);resultMsg.to(Device::CPU);
        const double logit=resultMsg.data()[0].real()+model.bias;
        const double prob=sigmoid(logit);
        const int pred=prob>=threshold?1:0;
        const int label=patient.label;
        tp+=(pred==1&&label==1);tn+=(pred==0&&label==0);
        fp+=(pred==1&&label==0);fn+=(pred==0&&label==1);
        const double t=elapsed(roundStart);timeSum+=t;timeSq+=t*t;
        scores.emplace_back(prob,label);
        output<<patient.id<<','<<patient.ahi<<','<<label<<','<<logit<<','<<prob<<','<<pred<<','<<t<<'\n';
        ++patientCount;
        if(patientCount%25==0||patientCount==patients.size())
            std::cout<<"PROGRESS "<<patientCount<<"/"<<patients.size()<<" avg_ms="<<timeSum/patientCount<<std::endl;
    }
    output.close();
    const auto ratio=[](double a,double b){return b>0?a/b:std::numeric_limits<double>::quiet_NaN();};
    const double accuracy=ratio(tp+tn,patientCount);
    const double sensitivity=ratio(tp,tp+fn);
    const double specificity=ratio(tn,tn+fp);
    const double precision=ratio(tp,tp+fp);
    const double f1=ratio(2.0*tp,2.0*tp+fp+fn);
    std::sort(scores.begin(),scores.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    double rankSum=0;size_t posCount=0,negCount=0;
    for(size_t i=0;i<scores.size();) {
        size_t j=i+1;while(j<scores.size()&&scores[j].first==scores[i].first)++j;
        const double avgRank=(double(i+1)+double(j))/2.0;
        for(size_t k=i;k<j;++k) {if(scores[k].second){rankSum+=avgRank;++posCount;}else ++negCount;}
        i=j;
    }
    const double auc=posCount&&negCount?(rankSum-double(posCount)*(posCount+1)/2.0)/(double(posCount)*negCount):std::numeric_limits<double>::quiet_NaN();
    const double meanMs=timeSum/patientCount;
    const double sdMs=std::sqrt(std::max(0.0,timeSq/patientCount-meanMs*meanMs));
    std::ofstream summary(root+"v5_metrics.csv");
    summary<<"metric,value\n"<<std::setprecision(15);
    summary<<"threshold,"<<threshold<<"\npatients,"<<patientCount<<"\nTP,"<<tp<<"\nTN,"<<tn<<"\nFP,"<<fp<<"\nFN,"<<fn
           <<"\naccuracy,"<<accuracy<<"\nsensitivity,"<<sensitivity<<"\nspecificity,"<<specificity
           <<"\nprecision,"<<precision<<"\nf1,"<<f1<<"\nroc_auc,"<<auc
           <<"\nmean_inference_ms,"<<meanMs<<"\nsd_inference_ms,"<<sdMs<<"\n";
    std::cout<<std::fixed<<std::setprecision(8)
             <<"\n========== V5 EVALUATION ==========\n"
             <<"patients="<<patientCount<<" threshold="<<threshold<<"\n"
             <<"TP="<<tp<<" TN="<<tn<<" FP="<<fp<<" FN="<<fn<<"\n"
             <<"accuracy="<<accuracy<<" sensitivity="<<sensitivity
             <<" specificity="<<specificity<<" precision="<<precision
             <<" f1="<<f1<<" roc_auc="<<auc<<"\n"
             <<"mean_inference_ms="<<meanMs<<" sd_inference_ms="<<sdMs<<"\n"
             <<"predictions="<<root<<"v5_predictions.csv\n"
             <<"metrics="<<root<<"v5_metrics.csv\n"
             <<"DISTRIBUTED CKKS V5 SUCCESS\n";

}

int main(int argc, char** argv) {
    try {
        if (argc < 2)
            throw std::runtime_error(
                "Usage: DistributedInference client [threshold] | hospital 0|1|2");

        const std::string root="/home/user/devkit/examples/heaan2-examples/my_example/";
        const std::string mode=argv[1];
        if(mode=="client") {
            const double threshold=argc>=3?std::stod(argv[2]):DEFAULT_THRESHOLD;
            if(!(threshold>0.0&&threshold<1.0)) throw std::runtime_error("Threshold must be in (0, 1)");
            runClient(loadClientModel(root+"client_model.txt"),threshold);
        }
        else if(mode=="hospital"&&argc>=3) {
            const int h=std::stoi(argv[2]);
            if(h<0||h>=3) throw std::runtime_error("Invalid hospital ID");
            runHospital(h,loadHospitalModel(root+"hospital"+std::to_string(h)+"_model.txt",h));
        } else throw std::runtime_error("Invalid arguments");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
