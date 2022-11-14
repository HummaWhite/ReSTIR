#pragma once

#include <device_launch_parameters.h>

#include "scene.h"

#if DENOISER_ENCODE_NORMAL
#  define ENCODE_NORM(x) Math::encodeNormalHemiOct32(x)
#  define DECODE_NORM(x) Math::decodeNormalHemiOct32(x)
#else
#  define ENCODE_NORM(x) x
#  define DECODE_NORM(x) x
#endif

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

    __device__ NormT* normal() const { return devNormal[frameIdx]; }
    __device__ NormT* lastNormal() const { return devNormal[frameIdx ^ 1]; }

    __device__ int* primId() const { return devPrimId[frameIdx]; }
    __device__ int* lastPrimId() const { return devPrimId[frameIdx ^ 1]; }

#if DENOISER_ENCODE_POSITION
    __device__ float* depth() const { return devDepth[frameIdx]; }
    __device__ float* lastDepth() const { return devDepth[frameIdx ^ 1]; }
#else
    __device__ glm::vec3* position() const { return devPosition[frameIdx]; }
    __device__ glm::vec3* lastPosition() const { return devPosition[frameIdx ^ 1]; }
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