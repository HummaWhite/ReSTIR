#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "sceneStructs.h"
#include "mathUtil.h"

#define NullPrimitive -1

struct AABB {
    __host__ __device__ AABB() = default;

    __host__ __device__ AABB(glm::vec3 pMin, glm::vec3 pMax) : pMin(pMin), pMax(pMax) {}

    AABB(glm::vec3 va, glm::vec3 vb, glm::vec3 vc) :
        pMin(glm::min(glm::min(va, vb), vc)), pMax(glm::max(glm::max(va, vb), vc)) {}

    __host__ __device__ AABB(const AABB& a, const AABB& b) :
        pMin(glm::min(a.pMin, b.pMin)), pMax(glm::min(a.pMax, b.pMax)) {}

    __host__ __device__ AABB operator () (glm::vec3 p) {
        return { glm::min(pMin, p), glm::max(pMax, p) };
    }

    __host__ __device__ AABB operator () (const AABB& rhs) {
        return { glm::min(pMin, rhs.pMin), glm::max(pMax, rhs.pMax) };
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "[AABB " << "pMin = " << vec3ToString(pMin);
        ss << ", pMax = " << vec3ToString(pMax);
        ss << ", center = " << vec3ToString(this->center()) << "]";
        return ss.str();
    }

    __host__ __device__ bool in(glm::vec3 p) const {
        return (p.x < pMax.x && p.y < pMax.y && p.z < pMax.z &&
            p.x > pMin.x && p.y > pMin.y && p.z > pMin.z);
    }

    __host__ __device__ glm::vec3 center() const {
        return (pMin + pMax) * .5f;
    }

    __host__ __device__ float surfaceArea() const {
        glm::vec3 size = pMax - pMin;
        return 2.f * (size.x * size.y + size.y * size.z + size.z * size.x);
    }

    /**
    * Returns 0 for X, 1 for Y, 2 for Z
    */
    __host__ __device__ int longestAxis() const {
        glm::vec3 size = pMax - pMin;
        if (size.x < size.y) {
            return size.y > size.z ? 1 : 2;
        }
        else {
            return size.x > size.z ? 0 : 2;
        }
    }

    __host__ __device__ bool getDistMinMax(float tMin1, float tMin2, float tMax1, float tMax2, float& tMin) {
        tMin = fminf(tMin1, tMin2);
        float tMax = fmaxf(tMax1, tMax2);
        return (tMax >= 0.f && tMax >= tMin);
    }

    __host__ __device__ bool getDistMaxMin(float tMin1, float tMin2, float tMax1, float tMax2, float& tMin) {
        tMin = fmaxf(tMin1, tMin2);
        float tMax = fminf(tMax1, tMax2);
        return (tMax >= 0.f && tMax >= tMin);
    }

    /**
    * Manually unrolled intersection test
    * This is tested 20% faster than other implementations
    */
    __host__ __device__ bool intersect(Ray ray, float& tMin) {
        const float Eps = 1e-6f;
        float tMax;
        glm::vec3 ori = ray.origin;
        glm::vec3 dir = ray.direction;

        if (glm::abs(dir.x) > 1.f - Eps) {
            if (Math::between(ori.y, pMin.y, pMax.y) && Math::between(ori.z, pMin.z, pMax.z)) {
                float dirInvX = 1.f / dir.x;
                float t1 = (pMin.x - ori.x) * dirInvX;
                float t2 = (pMax.x - ori.x) * dirInvX;
                return getDistMinMax(t1, t2, t1, t2, tMin);
            }
            else {
                return false;
            }
        }
        else if (glm::abs(dir.y) > 1.f - Eps) {
            if (Math::between(ori.z, pMin.z, pMax.z) && Math::between(ori.x, pMin.x, pMax.x)) {
                float dirInvY = 1.f / dir.y;
                float t1 = (pMin.y - ori.y) * dirInvY;
                float t2 = (pMax.y - ori.y) * dirInvY;
                return getDistMinMax(t1, t2, t1, t2, tMin);
            }
            else {
                return false;
            }
        }
        else if (glm::abs(dir.z) > 1.f - Eps) {
            if (Math::between(ori.x, pMin.x, pMax.x) && Math::between(ori.y, pMin.y, pMax.y)) {
                float dirInvZ = 1.f / dir.z;
                float t1 = (pMin.z - ori.z) * dirInvZ;
                float t2 = (pMax.z - ori.z) * dirInvZ;
                return getDistMinMax(t1, t2, t1, t2, tMin);
            }
            else {
                return false;
            }
        }
        glm::vec3 dirInv = 1.f / dir;
        glm::vec3 t1 = (pMin - ori) * dirInv;
        glm::vec3 t2 = (pMax - ori) * dirInv;

        glm::vec3 tNear = glm::min(t1, t2);
        glm::vec3 tFar = glm::max(t1, t2);

        glm::vec3 tDist = tFar - tNear;

        float yz = tFar.z - tNear.y;
        float zx = tFar.x - tNear.z;
        float xy = tFar.y - tNear.x;

        if (glm::abs(dir.x) < Eps && tDist.y + tDist.z > yz) {
            return getDistMaxMin(tNear.y, tNear.z, tFar.y, tFar.z, tMin);
        }

        if (glm::abs(dir.y) < Eps && tDist.z + tDist.x > zx) {
            return getDistMaxMin(tNear.z, tNear.x, tFar.z, tFar.x, tMin);
        }

        if (glm::abs(dir.z) < Eps && tDist.x + tDist.y > xy){
            return getDistMaxMin(tNear.x, tNear.y, tFar.x, tFar.y, tMin);
        }

        if (tDist.y + tDist.z > yz && tDist.z + tDist.x > zx && tDist.x + tDist.y > xy)
        {
            return getDistMaxMin(
                fmaxf(tNear.x, tNear.y), tNear.z, 
                fminf(tFar.x, tFar.y), tFar.z, tMin
            );
        }
        return false;
    }

    glm::vec3 pMin = glm::vec3(FLT_MAX);
    glm::vec3 pMax = glm::vec3(-FLT_MAX);
};

struct MTBVHNode {
    MTBVHNode() = default;
    MTBVHNode(int primId, int boxId, int next) :
        primitiveId(primId), boundingBoxId(boxId), nextNodeIfMiss(next) {}

    int primitiveId;
    int boundingBoxId;
    int nextNodeIfMiss;
};

class BVHBuilder {
private:
    struct NodeInfo {
        bool isLeaf;
        int primIdOrSize;
    };

    struct PrimInfo {
        int primId;
        AABB bound;
        glm::vec3 center;
    };

    struct BuildInfo {
        int offset;
        int start;
        int end;
    };

public:
    static int build(
        const std::vector<glm::vec3>& vertices, 
        std::vector<AABB>& boundingBoxes,
        std::vector<std::vector<MTBVHNode>>& BVHNodes);

private:
    static void buildMTBVH(
        const std::vector<AABB>& boundingBoxes,
        const std::vector<NodeInfo>& nodeInfo,
        int BVHSize,
        std::vector<std::vector<MTBVHNode>>& BVHNodes);
};