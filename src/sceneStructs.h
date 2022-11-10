#pragma once

#include <iomanip>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

struct Ray {
    __host__ __device__ glm::vec3 getPoint(float dist) {
        return origin + direction * dist;
    }

    glm::vec3 origin;
    glm::vec3 direction;
};

struct Camera {
    void update() {
        float yaw = glm::radians(rotation.x);
        float pitch = glm::radians(rotation.y);
        float roll = glm::radians(rotation.z);
        view.x = glm::cos(yaw) * glm::cos(pitch);
        view.z = glm::sin(yaw) * glm::cos(pitch);
        view.y = glm::sin(pitch);

        view = glm::normalize(view);
        right = glm::normalize(glm::cross(view, glm::vec3(0, 1, 0)));
        up = glm::normalize(glm::cross(right, view));
    }

    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDist;
    float tanFovY;
};

struct PrevBSDFSampleInfo {
    float BSDFPdf;
    bool deltaSample;
};

struct PathSegment {
    Ray ray;
    glm::vec3 throughput;
    glm::vec3 radiance;
    PrevBSDFSampleInfo prev;
    int pixelIndex;
    int remainingBounces;
};

struct RenderState {
    unsigned int iterations;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct Intersection {
    __device__ Intersection() {}

    __device__ Intersection(const Intersection& rhs) {
        *this = rhs;
    }

    __device__ void operator = (const Intersection& rhs) {
        primId = rhs.primId;
        matId = rhs.matId;
        pos = rhs.pos;
        norm = rhs.norm;
        uv = rhs.uv;
        wo = rhs.wo;
        prev = rhs.prev;
    }

    int primId;
    int matId;

    glm::vec3 pos;
    glm::vec3 norm;
    glm::vec2 uv;

    union {
        glm::vec3 wo;
        glm::vec3 prevPos;
    };

    PrevBSDFSampleInfo prev;
};