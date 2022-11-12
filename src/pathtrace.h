#pragma once

#include <device_launch_parameters.h>
#include <vector>
#include "scene.h"
#include "common.h"

#define ReservoirSize 64

__host__ __device__ static bool operator < (float x, glm::vec3 y) {
    return x < Math::luminance(y);
}

template<typename SampleT, typename WeightT>
struct Reservoir {
    __host__ __device__ void update(const SampleT& val, WeightT weight, float r) {
        sumWeight += weight;
        numSamples++;
        if (r < weight / sumWeight) {
            sampled = val;
        }
    }

    __host__ __device__ void merge(Reservoir rhs, WeightT g, float r) {
        int M0 = numSamples;
        update(rhs.sampled, rhs.sumWeight * g * static_cast<float>(rhs.numSamples), r);
        numSamples = M0 + rhs.numSamples;
    }

    __host__ __device__ void clear() {
        sumWeight = WeightT(0.f);
        numSamples = 0;
    }

    SampleT sampled = SampleT();
    WeightT sumWeight = WeightT(0.f);
    int numSamples = 0;
};

struct LightLiSample {
    glm::vec3 Li;
    glm::vec3 wi;
    float dist;
};

using DirectReservoir = Reservoir<LightLiSample, glm::vec3>;

void InitDataContainer(GuiDataContainer* guiData);

void copyImageToPBO(uchar4* devPBO, glm::vec3* devImage, int width, int height, int toneMapping, float scale = 1.f);
void copyImageToPBO(uchar4* devPBO, glm::vec2* devImage, int width, int height);
void copyImageToPBO(uchar4* devPBO, float* devImage, int width, int height);
void copyImageToPBO(uchar4* devPBO, int* devImage, int width, int height);

void pathTraceInit(Scene *scene);
void pathTraceFree();
void pathTrace(glm::vec3* devDirectIllum, glm::vec3* devIndirectIllum, int iter);

void ReSTIRDirect(glm::vec3* devDirectOutput, int iter, bool useReservoir);