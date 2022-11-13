#pragma once

#include <device_launch_parameters.h>
#include <vector>
#include "scene.h"
#include "common.h"

#define ReservoirSize 16

__host__ __device__ inline bool operator < (float x, glm::vec3 y) {
    //return x < Math::luminance(y);
    return x * x < glm::dot(y, y);
}

__host__ __device__ inline bool operator < (glm::vec3 x, glm::vec3 y) {
    //return Math::luminance(x) < Math::luminance(y);
    return glm::dot(x, x) < glm::dot(y, y);
}

template<typename SampleT>
struct Reservoir {
    __host__ __device__ Reservoir() : sample({}), sumWeight(1e-6f), resvWeight(0.f) {}

    __host__ __device__ void update(const SampleT& val, const glm::vec3& weight, float r) {
        sumWeight += weight;
        numSamples++;
        if (r < weight / sumWeight) {
            sample = val;
        }
    }

    __host__ __device__ void clear() {
        sumWeight = glm::vec3(1e-6f);
        resvWeight = glm::vec3(0.f);
        numSamples = 0;
    }

    __device__ glm::vec3 directPHat(const Intersection& intersec, const Material& material) const {
        return sample.Li * material.BSDF(intersec.norm, intersec.wo, sample.wi) * Math::satDot(intersec.norm, sample.wi);
    }

    __device__ void calcReservoirWeight(const Intersection& intersec, const Material& material) {
        resvWeight = sumWeight / (directPHat(intersec, material) * static_cast<float>(numSamples));
    }

    __device__ void merge(const Reservoir& rhs, const Intersection& intersec, const Material& material, float r) {
        glm::vec3 weight = directPHat(intersec, material) * resvWeight * static_cast<float>(numSamples);
        glm::vec3 rhsWeight = rhs.directPHat(intersec, material) * rhs.resvWeight * static_cast<float>(rhs.numSamples);
        sumWeight = weight + rhsWeight;

        if (r * rhsWeight < sumWeight) {
            sample = rhs.sample;
        }
        numSamples += rhs.numSamples;
        calcReservoirWeight(intersec, material);
    }

    SampleT sample = SampleT();
    int numSamples = 0;
    glm::vec3 sumWeight = glm::vec3(1e-6f);
    glm::vec3 resvWeight = glm::vec3(0.f);
};

struct LightLiSample {
    glm::vec3 Li;
    glm::vec3 wi;
    float dist;
};

using DirectReservoir = Reservoir<LightLiSample>;

void InitDataContainer(GuiDataContainer* guiData);

void copyImageToPBO(uchar4* devPBO, glm::vec3* devImage, int width, int height, int toneMapping, float scale = 1.f);
void copyImageToPBO(uchar4* devPBO, glm::vec2* devImage, int width, int height);
void copyImageToPBO(uchar4* devPBO, float* devImage, int width, int height);
void copyImageToPBO(uchar4* devPBO, int* devImage, int width, int height);

void pathTraceInit(Scene *scene);
void pathTraceFree();
void pathTrace(glm::vec3* devDirectIllum, glm::vec3* devIndirectIllum, int iter);

void ReSTIRDirect(glm::vec3* devDirectOutput, int iter, bool useReservoir);