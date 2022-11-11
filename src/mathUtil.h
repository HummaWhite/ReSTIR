#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define Pi 3.1415926535897932384626422832795028841971f
#define PiTwo 6.2831853071795864769252867665590057683943f
#define PiInv 1.f / Pi
#define OneThirdSqrt 0.5773502691896257645091487805019574556476f
#define EpsCmp 0.00001f

static std::string vec3ToString(const glm::vec3& vec) {
    std::stringstream ss;
    ss << "{ x = " << vec.x << ", y = " << vec.y << ", z = " << vec.z << " }";
    return ss.str();
}

__host__ __device__ inline int ceilDiv(int x, int y) {
    return (x + y - 1) / y;
}

namespace Math {
    bool epsilonCheck(float a, float b);
    glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);

    template<typename T>
    bool between(const T& x, const T& min, const T& max) {
        return x >= min && x <= max;
    }

    __host__ __device__ inline glm::vec3 HDRToLDR(glm::vec3 c) {
        return c / (c + 1.f) * 1.f;
    }

    __host__ __device__ inline glm::vec3 LDRToHDR(glm::vec3 c) {
        c /= 1.f;
        return c / (1.f - c + 1e-4f);
    }

    __host__ __device__ inline glm::vec2 encodeNormalHemiOct32(glm::vec3 n) {
        glm::vec2 p = glm::vec2(n) * (1.f / (glm::abs(n.x) + glm::abs(n.y) + n.z));
        return glm::vec2(p.x + p.y, p.x - p.y);
    }

    __host__ __device__ inline glm::vec3 decodeNormalHemiOct32(glm::vec2 n) {
        glm::vec2 temp = glm::vec2(n.x + n.y, n.x - n.y) * .5f;
        glm::vec3 v(temp, 1.f - glm::abs(temp.x) - glm::abs(temp.y));
        return glm::normalize(v);
    }

    __host__ __device__ inline bool hasNanOrInf(glm::vec3 v) {
        return isnan(v.x) || isnan(v.y) || isnan(v.z) || isinf(v.x) || isinf(v.y) || isinf(v.z);
    }

    __host__ __device__ inline float satDot(glm::vec3 a, glm::vec3 b) {
        return glm::max(glm::dot(a, b), 0.f);
    }

    __host__ __device__ inline float absDot(glm::vec3 a, glm::vec3 b) {
        return glm::abs(glm::dot(a, b));
    }

    __host__ __device__ inline float pow5(float x) {
        float x2 = x * x;
        return x2 * x2 * x;
    }

    __host__ __device__ inline float square(float x) {
        return x * x;
    }

    __host__ __device__ inline float powerHeuristic(float f, float g) {
        float f2 = f * f;
        return f2 / (f2 + g * g);
    }

    __host__ __device__ inline float triangleArea(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2) {
        return glm::length(glm::cross(v1 - v0, v2 - v0)) * .5f;
    }

    __host__ __device__ inline glm::vec3 triangleNormal(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2) {
        return glm::normalize(glm::cross(v1 - v0, v2 - v0));
    }

    __host__ __device__ static glm::vec3 sampleTriangleUniform(
        glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float ru, float rv) {
        float r = glm::sqrt(rv);
        float u = 1.f - r;
        float v = ru * r;
        return v1 * u + v2 * v + v0 * (1.f - u - v);
    }

    template<typename T>
    __host__ __device__ inline T calcFilmic(T c) {
        return (c * (c * 0.22f + 0.03f) + 0.002f) / (c * (c * 0.22f + 0.3f) + 0.06f) - 1.f / 30.f;
    }

    __host__ __device__ inline glm::vec3 filmic(glm::vec3 c) {
        return calcFilmic(c * 1.6f) / calcFilmic(11.2f);
    }

    __host__ __device__ inline glm::vec3 ACES(glm::vec3 color) {
        return (color * (color * 2.51f + 0.03f)) / (color * (color * 2.43f + 0.59f) + 0.14f);
    }

    __host__ __device__ inline glm::vec3 correctGamma(glm::vec3 color) {
        return glm::pow(color, glm::vec3(1.f / 2.2f));
    }

    __host__ __device__ inline float luminance(glm::vec3 color) {
        //const glm::vec3 T(.299f, .587f, .114f);
        const glm::vec3 T(.2126f, .7152f, .0722f);
        return glm::dot(color, T);
    }

    /**
    * Map a pair of evenly distributed [0, 1] coordinate to disc
    */
    __device__ static glm::vec2 toConcentricDisk(float x, float y) {
        float r = glm::sqrt(x);
        float theta = y * Pi * 2.0f;
        return glm::vec2(glm::cos(theta), glm::sin(theta)) * r;
    }

    __device__ static glm::vec3 toSphere(glm::vec2 v) {
        v *= glm::vec2(PiTwo, Pi);
        return glm::vec3(glm::cos(v.x) * glm::sin(v.y), glm::cos(v.y), glm::sin(v.x) * glm::sin(v.y));
    }

    __device__ static glm::vec2 toPlane(glm::vec3 v) {
        return glm::vec2(
            glm::fract(glm::atan(v.z, v.x) * PiInv * .5f + 1.f),
            glm::atan(glm::length(glm::vec2(v.x, v.z)), v.y) * PiInv
        );
    }

    __device__ static glm::mat3 localRefMatrix(glm::vec3 n) {
        glm::vec3 t = (glm::abs(n.y) > 0.9999f) ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
        glm::vec3 b = glm::normalize(glm::cross(n, t));
        t = glm::cross(b, n);
        return glm::mat3(t, b, n);
    }

    __device__ static glm::vec3 localToWorld(glm::vec3 n, glm::vec3 v) {
        return glm::normalize(localRefMatrix(n) * v);
    }

    __device__ static glm::vec3 sampleHemisphereCosine(glm::vec3 n, float rx, float ry) {
        glm::vec2 d = toConcentricDisk(rx, ry);
        float z = glm::sqrt(1.f - glm::dot(d, d));
        return localToWorld(n, glm::vec3(d, z));
    }

    __device__ static bool refract(glm::vec3 n, glm::vec3 wi, float ior, glm::vec3& wt) {
        float cosIn = glm::dot(n, wi);
        if (cosIn < 0) {
            ior = 1.f / ior;
        }
        float sin2In = glm::max(0.f, 1.f - cosIn * cosIn);
        float sin2Tr = sin2In / (ior * ior);

        if (sin2Tr >= 1.f) {
            return false;
        }
        float cosTr = glm::sqrt(1.f - sin2Tr);
        if (cosIn < 0) {
            cosTr = -cosTr;
        }
        wt = glm::normalize(-wi / ior + n * (cosIn / ior - cosTr));
        return true;
    }

    __device__ inline float pdfAreaToSolidAngle(float pdf, glm::vec3 x, glm::vec3 y, glm::vec3 ny) {
        glm::vec3 yx = x - y;
        return pdf * glm::dot(yx, yx) / absDot(ny, glm::normalize(yx));
    }

    /**
    * Handy-dandy hash function that provides seeds for random number generation.
    */
    __host__ __device__ inline unsigned int utilhash(unsigned int a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }
}