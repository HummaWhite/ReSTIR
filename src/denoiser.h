#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "common.h"
#include "scene.h"
#include "pathtrace.h"

struct GBuffer {
#if DENOISER_ENCODE_NORMAL
    using NormT = glm::vec2;
#else
    using NormT = glm::vec3;
#endif

    GBuffer() = default;

    void create(int width, int height);
    void destroy();
    void render(DevScene* scene, const Camera& cam);
    void update(const Camera& cam);

    __device__ NormT* normal() { return devNormal[frameIdx]; }
    __device__ NormT* lastNormal() { return devNormal[frameIdx ^ 1]; }

    __device__ int* primId() { return devPrimId[frameIdx]; }
    __device__ int* lastPrimId() { return devPrimId[frameIdx ^ 1]; }

#if DENOISER_ENCODE_POSITION
    __device__ float* depth() { return devDepth[frameIdx]; }
    __device__ float* lastDepth() { return devDepth[frameIdx ^ 1]; }
#else
    __device__ glm::vec3* position() { return devPosition[frameIdx]; }
    __device__ glm::vec3* lastPosition() { return devPosition[frameIdx ^ 1]; }
#endif

    glm::vec3* devAlbedo = nullptr;

    int* devMotion = nullptr;
    NormT* devNormal[2] = { nullptr };

#if DENOISER_ENCODE_POSITION
    float* devDepth[2] = { nullptr };
#else
    glm::vec3* devPosition[2] = { nullptr };
#endif
    int* devPrimId[2] = { nullptr };
    int frameIdx = 0;

    Camera lastCamera;
    int width;
    int height;
};

struct EAWaveletFilter {
    EAWaveletFilter() = default;

    EAWaveletFilter(int width, int height, float sigLumin, float sigNormal, float sigDepth) :
        width(width), height(height), sigLumin(sigLumin), sigNormal(sigNormal), sigDepth(sigDepth) {}

    void filter(glm::vec3* devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam, int level);
    void filter(glm::vec3* devColorOut, glm::vec3* devColorIn, float* devVarianceOut, float* devVarianceIn,
        float* devFilteredVar, const GBuffer& gBuffer, const Camera& cam, int level);

    float sigLumin;
    float sigNormal;
    float sigDepth;

    int width = 0;
    int height = 0;
};

struct LeveledEAWFilter {
    LeveledEAWFilter() = default;
    void create(int width, int height, int level);
    void destroy();

    void filter(glm::vec3*& devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam);

    EAWaveletFilter waveletFilter;
    int level = 0;
    glm::vec3* devTempImg = nullptr;
};

struct SpatioTemporalFilter {
    SpatioTemporalFilter() = default;
    void create(int width, int height, int level);
    void destroy();

    void temporalAccumulate(glm::vec3* devColorIn, const GBuffer& gBuffer);
    void estimateVariance();
    void filterVariance();

    void filter(glm::vec3*& devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam);

    void nextFrame();

    EAWaveletFilter waveletFilter;
    int level = 0;

    glm::vec3* devAccumColor[2] = { nullptr };
    glm::vec3* devAccumMoment[2] = { nullptr };
    float* devVariance = nullptr;
    bool firstTime = true;

    glm::vec3* devTempColor = nullptr;
    float* devTempVariance = nullptr;
    float* devFilteredVariance = nullptr;
    int frameIdx = 0;
};

void modulateAlbedo(glm::vec3* devImage, const GBuffer& gBuffer);
void addImage(glm::vec3* devImage, glm::vec3* devIn, int width, int height);
void addImage(glm::vec3* devOut, glm::vec3* devIn1, glm::vec3* devIn2, int width, int height);