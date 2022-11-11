#pragma once

#include <iomanip>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "common.h"

struct Ray {
    __host__ __device__ glm::vec3 getPoint(float dist) {
        return origin + direction * dist;
    }

    glm::vec3 origin;
    glm::vec3 direction;
};

struct Camera {
    __device__ glm::vec2 getRasterUV(glm::vec3 pos) {
        /*glm::vec4 projected = viewProjection * glm::vec4(pos, 1.f);
        projected /= projected.w;
        projected = -projected;
        projected = projected * .5f + .5f;
        return glm::vec2(projected.x, projected.y);*/

        glm::vec3 dir = glm::normalize(pos - position);
        float d = 1.f / glm::dot(dir, view);

        glm::vec3 p = rotationMatInv * (dir * d);
        float aspect = float(resolution.x) / resolution.y;
        float tanFovY = glm::tan(glm::radians(fov.y));

        p /= glm::vec3(glm::vec2(aspect, 1.f) * tanFovY, 1.f);
        glm::vec2 ndc(p);
        ndc = -ndc;
        return ndc * .5f + .5f;
    }

    __device__ glm::ivec2 getRasterCoord(glm::vec3 pos) {
        glm::vec2 ndc = getRasterUV(pos);
        int ix = glm::clamp(int((resolution.x - 1) * ndc.x + .5f), 0, resolution.x - 1);
        int iy = glm::clamp(int((resolution.y - 1) * ndc.y + .5f), 0, resolution.y - 1);
        return { ix, iy };
    }

    __device__ glm::vec3 getPosition(int x, int y, float dist) {
        float aspect = float(resolution.x) / resolution.y;
        float tanFovY = glm::tan(glm::radians(fov.y));
        glm::vec2 pixelSize = 1.f / glm::vec2(resolution);
        glm::vec2 scr = glm::vec2(x, y) * pixelSize;
        glm::vec2 ruv = scr + pixelSize * .5f;
        ruv = 1.f - ruv * 2.f;

        glm::vec2 pAperture(0.f);
        glm::vec3 pLens = glm::vec3(pAperture * lensRadius, 0.f);
        glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * focalDist;
        glm::vec3 dir = pFocusPlane - pLens;

        dir = glm::normalize(glm::mat3(right, up, view) * dir);
        glm::vec3 ori = position + right * pLens.x + up * pLens.y;
        return ori + dir * dist;
    }

    /*
    * Antialiasing and physically based camera (lens effect)
    */
    __device__ Ray sample(int x, int y, glm::vec4 r) {
        Ray ray;
        float aspect = float(resolution.x) / resolution.y;
        float tanFovY = glm::tan(glm::radians(fov.y));
        glm::vec2 pixelSize = 1.f / glm::vec2(resolution);
        glm::vec2 scr = glm::vec2(x, y) * pixelSize;
        glm::vec2 ruv = scr + pixelSize * glm::vec2(r.x, r.y);
        ruv = 1.f - ruv * 2.f;

        glm::vec2 pAperture(0.f);
        glm::vec3 pLens = glm::vec3(pAperture * lensRadius, 0.f);
        glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * focalDist;
        glm::vec3 dir = pFocusPlane - pLens;

        ray.direction = glm::normalize(glm::mat3(right, up, view) * dir);
        ray.origin = position + right * pLens.x + up * pLens.y;
        return ray;
    }

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
        rotationMatInv = glm::inverse(glm::mat3(right, up, view));

        viewProjection = projMatrix() * viewMatrix();
    }

    glm::mat4 viewMatrix() const {
        return glm::lookAt(position, position + view, up);
    }

    glm::mat4 projMatrix() const {
        float aspect = static_cast<float>(resolution.x) / resolution.y;
        return glm::perspective(glm::radians(fov.y * 2.f), aspect, .01f, 1000.f);
    }

    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    glm::mat3 rotationMatInv;
    glm::mat4 viewProjection;
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
    glm::vec3 directIllum;
    glm::vec3 indirectIllum;
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