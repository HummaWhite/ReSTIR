#pragma once

#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaUtil.h"
#include "intersections.h"
#include "sceneStructs.h"
#include "material.h"
#include "image.h"
#include "bvh.h"
#include "sampler.h"
#include "common.h"

struct MeshData {
    void clear() {
        vertices.clear();
        normals.clear();
        texcoords.clear();
    }

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
};

struct ModelInstance {
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;

    glm::mat4 transform;
    glm::mat4 transfInv;
    glm::mat3 normalMat;

    int materialId;
    MeshData* meshData;
};

class Resource {
public:
    static MeshData* loadOBJMesh(const std::string& filename);
    static MeshData* loadGLTFMesh(const std::string& filename);
    static MeshData* loadModelMeshData(const std::string& filename);
    static Image* loadTexture(const std::string& filename);

    static void clear();

public:
    static std::map<std::string, MeshData*> meshDataPool;
    static std::map<std::string, Image*> texturePool;
};

class Scene;

struct DevScene {
    void create(const Scene& scene);
    void destroy();

    __device__ glm::vec3 proceduralTexture(glm::vec2 uv) {
        thrust::default_random_engine rng(int(uv.x * 1024) * 1024 + int(uv.y * 1024));
        float rx = thrust::uniform_real_distribution<float>(0.f, 1.f)(rng);
        float ry = thrust::uniform_real_distribution<float>(0.f, 1.f)(rng);

        float f = (glm::sin(uv.x * 10.f * PiTwo + rx * PiTwo) + 1.f) * .5f;
        float g = (glm::sin(uv.y * 10.f * PiTwo + ry * PiTwo) + 1.f) * .5f;
        return glm::vec3(f * g);
    }

    __device__ Material getTexturedMaterialAndSurface(Intersection& intersec) {
        Material mat = materials[intersec.matId];
        if (mat.baseColorMapId != NullTextureId) {
            mat.baseColor = mat.baseColorMapId == ProceduralTexId ?
                proceduralTexture(intersec.uv) : textures[mat.baseColorMapId].linearSample(intersec.uv);
        }

        if (mat.metallicMapId > NullTextureId) {
            mat.metallic = textures[mat.metallicMapId].linearSample(intersec.uv).r;
        }

        if (mat.roughnessMapId > NullTextureId) {
            mat.roughness = textures[mat.roughnessMapId].linearSample(intersec.uv).r;
        }

        if (mat.normalMapId != NullTextureId) {
            glm::vec3 mapped = textures[mat.normalMapId].linearSample(intersec.uv);
            glm::vec3 localNorm = glm::normalize(glm::vec3(mapped.x, mapped.y, mapped.z) * 1.f - 0.5f);
            intersec.norm = Math::localToWorld(intersec.norm, localNorm);
        }
        return mat;
    }

    __device__ int getMTBVHId(glm::vec3 dir) {
        glm::vec3 absDir = glm::abs(dir);
        if (absDir.x > absDir.y) {
            if (absDir.x > absDir.z) {
                return dir.x > 0 ? 0 : 1;
            }
            else {
                return dir.z > 0 ? 4 : 5;
            }
        }
        else {
            if (absDir.y > absDir.z) {
                return dir.y > 0 ? 2 : 3;
            }
            else {
                return dir.z > 0 ? 4 : 5;
            }
        }
    }

    __device__ float getPrimitiveArea(int primId) {
        glm::vec3 v0 = vertices[primId * 3 + 0];
        glm::vec3 v1 = vertices[primId * 3 + 1];
        glm::vec3 v2 = vertices[primId * 3 + 2];
        return glm::length(glm::cross(v1 - v0, v2 - v0)) * .5f;
    }

    __device__ glm::vec3 getPrimitivePlainNormal(int primId) {
        glm::vec3 v0 = vertices[primId * 3 + 0];
        glm::vec3 v1 = vertices[primId * 3 + 1];
        glm::vec3 v2 = vertices[primId * 3 + 2];
        return glm::normalize(glm::cross(v1 - v0, v2 - v0));
    }

    __device__ void getIntersecGeomInfo(int primId, glm::vec2 bary, Intersection& intersec) {
        glm::vec3 va = vertices[primId * 3 + 0];
        glm::vec3 vb = vertices[primId * 3 + 1];
        glm::vec3 vc = vertices[primId * 3 + 2];

        glm::vec3 na = normals[primId * 3 + 0];
        glm::vec3 nb = normals[primId * 3 + 1];
        glm::vec3 nc = normals[primId * 3 + 2];

        glm::vec2 ta = texcoords[primId * 3 + 0];
        glm::vec2 tb = texcoords[primId * 3 + 1];
        glm::vec2 tc = texcoords[primId * 3 + 2];

        intersec.pos = vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
        intersec.norm = glm::normalize(nb * bary.x + nc * bary.y + na * (1.f - bary.x - bary.y));
        intersec.uv = tb * bary.x + tc * bary.y + ta * (1.f - bary.x - bary.y);
    }

    __device__ bool intersectPrimitive(int primId, Ray ray, float& dist, glm::vec2& bary) {
        glm::vec3 va = vertices[primId * 3 + 0];
        glm::vec3 vb = vertices[primId * 3 + 1];
        glm::vec3 vc = vertices[primId * 3 + 2];

        if (!intersectTriangle(ray, va, vb, vc, bary, dist)) {
            return false;
        }
        glm::vec3 hitPoint = vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
        return true;
    }

    __device__ bool intersectPrimitive(int primId, Ray ray, float distRange) {
        glm::vec3 va = vertices[primId * 3 + 0];
        glm::vec3 vb = vertices[primId * 3 + 1];
        glm::vec3 vc = vertices[primId * 3 + 2];
        glm::vec2 bary;
        float dist;
        bool hit = intersectTriangle(ray, va, vb, vc, bary, dist);
        return (hit && dist < distRange);
    }

    __device__ bool intersectPrimitiveDetailed(int primId, Ray ray, Intersection& intersec) {
        glm::vec3 va = vertices[primId * 3 + 0];
        glm::vec3 vb = vertices[primId * 3 + 1];
        glm::vec3 vc = vertices[primId * 3 + 2];
        float dist;
        glm::vec2 bary;

        if (!intersectTriangle(ray, va, vb, vc, bary, dist)) {
            return false;
        }

        glm::vec3 na = normals[primId * 3 + 0];
        glm::vec3 nb = normals[primId * 3 + 1];
        glm::vec3 nc = normals[primId * 3 + 2];

        glm::vec2 ta = texcoords[primId * 3 + 0];
        glm::vec2 tb = texcoords[primId * 3 + 1];
        glm::vec2 tc = texcoords[primId * 3 + 2];

        intersec.pos = vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
        intersec.norm = glm::normalize(nb * bary.x + nc * bary.y + na * (1.f - bary.x - bary.y));
        intersec.uv = tb * bary.x + tc * bary.y + ta * (1.f - bary.x - bary.y);
        return true;
    }

    __device__ void naiveIntersect(Ray ray, Intersection& intersec) {
        float closestDist = FLT_MAX;
        int closestPrimId = NullPrimitive;
        glm::vec2 closestBary;

        for (int i = 0; i < (BVHSize + 1) / 2; i++) {
            float dist;
            glm::vec2 bary;
            bool hit = intersectPrimitive(i, ray, dist, bary);

            if (hit && dist < closestDist) {
                closestDist = dist;
                closestBary = bary;
                closestPrimId = i;
            }
        }

        if (closestPrimId != NullPrimitive) {
            getIntersecGeomInfo(closestPrimId, closestBary, intersec);
            intersec.primId = closestPrimId;
            intersec.matId = materialIds[closestPrimId];
        }
        else {
            intersec.primId = NullPrimitive;
        }
    }

    __device__ bool naiveTestOcclusion(glm::vec3 x, glm::vec3 y) {
        const float Eps = 1e-4f;

        glm::vec3 dir = y - x;
        float dist = glm::length(dir);
        dir /= dist;
        dist -= Eps;

        Ray ray = makeOffsetedRay(x, dir);

        for (int i = 0; i < (BVHSize + 1) / 2; i++) {
            if (intersectPrimitive(i, ray, dist)) {
                return true;
            }
        }
        return false;
    }

    __device__ void intersect(Ray ray, Intersection& intersec) {
        float closestDist = FLT_MAX;
        int closestPrimId = NullPrimitive;
        glm::vec2 closestBary;

        MTBVHNode* nodes = BVHNodes[getMTBVHId(-ray.direction)];
        int node = 0;

        while (node != BVHSize) {
            AABB& bound = boundingBoxes[nodes[node].boundingBoxId];
            float boundDist;
            bool boundHit = bound.intersect(ray, boundDist);

            // Only intersect a primitive if its bounding box is hit and
            // that box is closer than previous hit record
            if (boundHit && boundDist < closestDist) {
                int primId = nodes[node].primitiveId;
                if (primId != NullPrimitive) {
                    float dist;
                    glm::vec2 bary;
                    bool hit = intersectPrimitive(primId, ray, dist, bary);

                    if (hit && dist < closestDist) {
                        closestDist = dist;
                        closestBary = bary;
                        closestPrimId = primId;
                    }
                }
                node++;
            }
            else {
                node = nodes[node].nextNodeIfMiss;
            }
        }
        if (closestPrimId != NullPrimitive) {
            getIntersecGeomInfo(closestPrimId, closestBary, intersec);
            intersec.matId = materialIds[closestPrimId];
        }
        intersec.primId = closestPrimId;
    }

    __device__ bool testOcclusion(glm::vec3 x, glm::vec3 y) {
        const float Eps = 1e-4f;

        glm::vec3 dir = y - x;
        float dist = glm::length(dir);
        dir /= dist;
        dist -= Eps;

        Ray ray = makeOffsetedRay(x, dir);

        MTBVHNode* nodes = BVHNodes[getMTBVHId(-ray.direction)];
        int node = 0;
        while (node != BVHSize) {
            AABB& bound = boundingBoxes[nodes[node].boundingBoxId];
            float boundDist;
            bool boundHit = bound.intersect(ray, boundDist);

            if (boundHit && boundDist < dist) {
                int primId = nodes[node].primitiveId;
                if (primId != NullPrimitive) {
                    if (intersectPrimitive(primId, ray, dist)) {
                        return true;
                    }
                }
                node++;
            }
            else {
                node = nodes[node].nextNodeIfMiss;
            }
        }
        return false;
    }

    __device__ void visualizedIntersect(Ray ray, Intersection& intersec) {
        float closestDist = FLT_MAX;
        int closestPrimId = NullPrimitive;
        glm::vec2 closestBary;

        MTBVHNode* nodes = BVHNodes[getMTBVHId(-ray.direction)];
        int node = 0;
        int maxDepth = 0;

        while (node != BVHSize) {
            AABB& bound = boundingBoxes[nodes[node].boundingBoxId];
            float boundDist;
            bool boundHit = bound.intersect(ray, boundDist);

            // Only intersect a primitive if its bounding box is hit and
            // that box is closer than previous hit record
            if (boundHit && boundDist < closestDist) {
                int primId = nodes[node].primitiveId;
                if (primId != NullPrimitive) {
                    float dist;
                    glm::vec2 bary;
                    bool hit = intersectPrimitive(primId, ray, dist, bary);

                    if (hit && dist < closestDist) {
                        closestDist = dist;
                        closestBary = bary;
                        closestPrimId = primId;
                        //maxDepth += 1.f;
                    }
                }
                node++;
                maxDepth += 1.f;
            }
            else {
                node = nodes[node].nextNodeIfMiss;
            }
        }
        intersec.primId = maxDepth;
    }

    __device__ float environmentMapPdf(glm::vec3 w) {
        glm::vec3 radiance = envMap->linearSample(Math::toPlane(w));
        return Math::luminance(radiance) * sumLightPowerInv *
            envMap->width * envMap->height * .5f;// *glm::sqrt(glm::max(1.f - w.z * w.z, FLT_EPSILON));
    }

    __device__ float sampleEnvironmentMap(glm::vec3 pos, glm::vec2 r, glm::vec3& radiance, glm::vec3& wi) {
        int pixId = envMapSampler.sample(r.x, r.y);
        
        int y = pixId / envMap->width;
        int x = pixId - y * envMap->width;

        radiance = envMap->devData[pixId];
        wi = Math::toSphere(glm::vec2((.5f + x) / envMap->width, (.5f + y) / envMap->height));

#if BVH_DISABLE
        bool occ = naiveTestOcclusion(pos, pos + wi * (FLT_MAX * .01f));
#else
        bool occ = testOcclusion(pos, pos + wi * 1e6f);
#endif
        if (occ) {
            return InvalidPdf;
        }
        
        return Math::luminance(radiance) * sumLightPowerInv *
            envMap->width * envMap->height * PiInv * PiInv * .5f;
    }

    /**
    * Returns solid angle probability
    */
    __device__ float sampleDirectLight(glm::vec3 pos, glm::vec4 r, glm::vec3& radiance, glm::vec3& wi) {
        if (lightSampler.length == 0) {
            return InvalidPdf;
        }
        int lightId = lightSampler.sample(r.x, r.y);

        if (lightId == lightSampler.length - 1 && envMapSampler.length != 0) {
            return sampleEnvironmentMap(pos, glm::vec2(r.z, r.w), radiance, wi);
        }
        int primId = lightPrimIds[lightId];

        glm::vec3 v0 = vertices[primId * 3 + 0];
        glm::vec3 v1 = vertices[primId * 3 + 1];
        glm::vec3 v2 = vertices[primId * 3 + 2];
        glm::vec3 sampled = Math::sampleTriangleUniform(v0, v1, v2, r.z, r.w);

#if BVH_DISABLE
        bool occ = naiveTestOcclusion(pos, sampled);
#else
        bool occ = testOcclusion(pos, sampled);
#endif
        if (occ) {
            return InvalidPdf;
        }
        glm::vec3 normal = Math::triangleNormal(v0, v1, v2);
        glm::vec3 posToSampled = sampled - pos;

#if SCENE_LIGHT_SINGLE_SIDED
        if (glm::dot(normal, posToSampled) > 0.f) {
            return InvalidPdf;
        }
#endif
        radiance = lightUnitRadiance[lightId];
        wi = glm::normalize(posToSampled);
        return Math::pdfAreaToSolidAngle(Math::luminance(radiance) * sumLightPowerInv, pos, sampled, normal);
    }

    glm::vec3* vertices = nullptr;
    glm::vec3* normals = nullptr;
    glm::vec2* texcoords = nullptr;
    AABB* boundingBoxes = nullptr;
    MTBVHNode* BVHNodes[6] = { nullptr };
    int BVHSize;

    int* materialIds = nullptr;
    Material* materials = nullptr;
    glm::vec3* textureData = nullptr;
    DevTextureObj* textures = nullptr;

    int* lightPrimIds = nullptr;
    glm::vec3* lightUnitRadiance = nullptr;
    DevDiscreteSampler1D<float> lightSampler;
    float sumLightPowerInv;
    DevTextureObj* envMap = nullptr;
    DevDiscreteSampler1D<float> envMapSampler;

    uint32_t* sampleSequence = nullptr;
};

class Scene {
public:
    friend struct DevScene;

    Scene(const std::string& filename);
    ~Scene();

    void buildDevData();
    void clear();

private:
    void createLightSampler();

    void loadModel(const std::string& objectId);
    void loadMaterial(const std::string& materialId);
    void loadCamera();

    int addMaterial(const Material& material);
    int addTexture(const std::string& filename);

public:
    RenderState state;
    std::vector<ModelInstance> modelInstances;
    std::vector<Image*> textures;
    std::map<Image*, int> textureMap;
    std::vector<Material> materials;
    std::map<std::string, int> materialMap;
    std::vector<int> materialIds;
    int BVHSize;
    std::vector<AABB> boundingBoxes;
    std::vector<std::vector<MTBVHNode>> BVHNodes;
    MeshData meshData;

    std::vector<int> lightPrimIds;
    std::vector<float> lightPower;
    std::vector<glm::vec3> lightUnitRadiance;
    DiscreteSampler1D<float> lightSampler;
    int numLightPrims = 0;
    DiscreteSampler1D<float> envMapSampler;
    int envMapTexId = NullTextureId;

    DevScene hstScene;
    DevScene* devScene = nullptr;

    Camera camera;

private:
    std::ifstream fpIn;
};
