#pragma once

#include "../examples.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>

using namespace heaan;

// 학습(dataset.MODEL_FEATURES)과 같은 순서의 16개 피처.
// 0~10: SHHS CSV / 워치 수면·프로필·앱 입력, 11~15: SpO2 1분 피처 (oximetry_features.py)
constexpr std::size_t NUM_HEALTH_FEATURES = 16;

enum class HealthFeature : std::size_t {
    AVGSAT = 0,
    AVG_HR,
    SLPPRDP,
    SLPEFFP,
    TIMES34P,
    AGE,
    GENDER,
    BMI,
    WASO,
    NECK20,
    ESS_S1,
    ODI_1MIN,
    PCT_LT90,
    MIN_SPO2,
    DESAT_DROP_1MIN,
    DESAT_MIN_FRAC
};

using HealthFeatureVector =
    std::array<Real, NUM_HEALTH_FEATURES>;

inline Message healthFeaturesToMessage(
    const HealthFeatureVector &features,
    u32 log_slots) {

    const std::size_t slot_count =
        static_cast<std::size_t>(1ULL << log_slots);

    if (slot_count < NUM_HEALTH_FEATURES) {
        throw std::runtime_error(
            "Not enough CKKS slots for health features");
    }

    // Message is first created and filled on CPU.
    Message msg(log_slots);

    Complex *msg_ptr = msg.data();

    for (std::size_t i = 0; i < slot_count; ++i) {
        msg_ptr[i] = Complex(0.0, 0.0);
    }

    for (std::size_t i = 0; i < NUM_HEALTH_FEATURES; ++i) {
        msg_ptr[i] = Complex(features[i], 0.0);
    }

    return msg;
}

inline void printHealthFeatures(
    const HealthFeatureVector &features) {

    static constexpr const char *names[NUM_HEALTH_FEATURES] = {
        "avgsat",
        "avg_hr",
        "slpprdp",
        "slpeffp",
        "times34p",
        "age",
        "gender",
        "bmi",
        "waso",
        "neck20",
        "ess_s1",
        "odi_1min",
        "pct_lt90",
        "min_spo2",
        "desat_drop_1min",
        "desat_min_frac"
    };

    std::cout << "===== Health Feature Vector =====\n";

    for (std::size_t i = 0; i < NUM_HEALTH_FEATURES; ++i) {
        std::cout
            << "[" << i << "] "
            << names[i]
            << " = "
            << features[i]
            << '\n';
    }

    std::cout << "=================================\n";
}
