#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "mathUtil.h"

__host__ __device__ inline Ray makeRay(glm::vec3 ori, glm::vec3 dir) {
    return { ori, dir };
}

__host__ __device__ inline Ray makeOffsetedRay(glm::vec3 ori, glm::vec3 dir) {
    return { ori + dir * 1e-5f, dir };
}

__host__ __device__ static bool intersectTriangle(Ray ray, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
    glm::vec2& bary, float& dist
) {
    glm::vec3 e01 = v1 - v0;
    glm::vec3 e02 = v2 - v0;
    glm::vec3 ori = ray.origin;
    glm::vec3 dir = ray.direction;
    glm::vec3 p = glm::cross(dir, e02);

    float det = glm::dot(p, e01);

    if (glm::abs(det) < FLT_EPSILON) {
        return false;
    }
    glm::vec3 v0ToOri = ori - v0;

    if (det < 0.f) {
        det = -det;
        v0ToOri = -v0ToOri;
    }

    bary.x = glm::dot(v0ToOri, p);

    if (bary.x < 0.f || bary.x > det) {
        return false;
    }
    glm::vec3 perp = glm::cross(v0ToOri, e01);
    bary.y = glm::dot(dir, perp);

    if (bary.y < 0.f || bary.x + bary.y > det) {
        return false;
    }
    
    float detInv = 1.f / det;
    dist = glm::dot(e02, perp) * detInv;
    bary *= detInv;
    return dist > 0.f;
}