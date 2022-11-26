#pragma once

#include <device_launch_parameters.h>
#include "scene.h"
#include "gbuffer.h"

struct DirectLiSample {
    glm::vec3 Li;
    glm::vec3 wi;
    float dist;
};

struct IndirectLiSample {
    __host__ __device__ IndirectLiSample() : Lo(0.f) {}

    __host__ __device__ bool invalid() const {
        return Math::luminance(Lo) < 1e-8f;
    }

    __host__ __device__ glm::vec3 wi() const {
        return glm::normalize(xs - xv);
    }

    glm::vec3 Lo;
    glm::vec3 xv, nv;
    glm::vec3 xs, ns;
};

template<typename SampleT>
struct Reservoir {
    __host__ __device__ Reservoir() = default;

    static __host__ __device__ float toScalar(glm::vec3 x) {
        return Math::luminance(x);
        //return glm::length(x);
    }

    __host__ __device__ void update(const SampleT& newSample, float newWeight, float r) {
        weight += newWeight;
        numSamples++;
        if (r * weight < newWeight) {
            sample = newSample;
        }
    }

    __host__ __device__ void clear() {
        weight = 0.f;
        numSamples = 0;
    }

    __device__ bool invalid() {
        return Math::isNanOrInf(weight) || weight < 0.f;
    }

    __device__ void checkValidity() {
        if (invalid()) {
            clear();
        }
    }

    __device__ void merge(const Reservoir& rhs, float r) {
        weight += rhs.weight;
        numSamples += rhs.numSamples;

        if (r * weight < rhs.weight) {
            sample = rhs.sample;
        }
    }

    __device__ void clampedMerge(Reservoir rhs, int threshold, float r) {
        int clamp = threshold - numSamples;
        if (rhs.numSamples > clamp) {
            rhs.weight = static_cast<float>(clamp) / rhs.numSamples;
            rhs.numSamples = clamp;
        }
        merge(rhs, r);
    }

    template<int M>
    __device__ void clamp() {
        static_assert(M > 0, "M <= 0");
        if (numSamples > M) {
            weight *= static_cast<float>(M) / numSamples;
            numSamples = M;
        }
    }

    __device__ void clamp(int val) {
        if (numSamples > val) {
            weight *= static_cast<float>(val) / numSamples;
            numSamples = val;
        }
    }

    template<int M>
    __device__ void preClampedMerge(Reservoir rhs, float r) {
        static_assert(M > 0, "M <= 0");
        if (numSamples > 0) {
            rhs.clamp((M - 1) * numSamples);
        }
        merge(rhs, r);
    }

    template<int M>
    __device__ void postClampedMerge(Reservoir rhs, float r) {
        static_assert(M > 0, "M <= 0");
        int curNumSample = numSamples;
        merge(rhs, r);
        if (numSamples > 0 && curNumSample > 0) {
            clamp(M * curNumSample);
        }
    }

    SampleT sample = SampleT();
    int numSamples = 0;
    float weight = 0.f;
};

template<typename T>
__device__ Reservoir<T> mergeReservoir(const Reservoir<T>& a, const Reservoir<T>& b, glm::vec2 r) {
    Reservoir<T> reservoir;
    reservoir.update(a.sample, a.weight, r.x);
    reservoir.update(b.sample, b.weight, r.y);
    reservoir.numSamples = a.numSamples + b.numSamples;
    return reservoir;
}

void ReSTIRInit();
void ReSTIRFree();
void ReSTIRReset();

void ReSTIRDirect(glm::vec3* devDirectIllum, int iter, const GBuffer& gBuffer);
void ReSTIRIndirect(glm::vec3* devIndirectIllum, int iter, const GBuffer& gBuffer);