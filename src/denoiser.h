#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "common.h"
#include "scene.h"
#include "gbuffer.h"
#include "pathtrace.h"

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