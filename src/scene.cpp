#include "scene.h"
#include "common.h"
#include "stb_image.h"
#include <iostream>
#include <cstring>
#include <stack>
#include <map>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <tiny_obj_loader.h>
#include <tiny_gltf.h>

std::map<std::string, int> MaterialTypeTokenMap = {
    { "Lambertian", Material::Type::Lambertian },
    { "MetallicWorkflow", Material::Type::MetallicWorkflow },
    { "Dielectric", Material::Type::Dielectric },
    { "Light", Material::Type::Light }
};

std::map<std::string, MeshData*> Resource::meshDataPool;
std::map<std::string, Image*> Resource::texturePool;

MeshData* Resource::loadOBJMesh(const std::string& filename) {
    auto find = meshDataPool.find(filename);
    if (find != meshDataPool.end()) {
        return find->second;
    }
    auto model = new MeshData;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::string warn, err;

    std::cout << "\t\t[Model loading " << filename << " ...]" << std::endl;
    if (!tinyobj::LoadObj(&attrib, &shapes, nullptr, &warn, &err, filename.c_str())) {
        std::cout << "\t\t\t[Fail Error msg " << err << "]" << std::endl;
        return nullptr;
    }
    bool hasTexcoord = !attrib.texcoords.empty();

    for (const auto& shape : shapes) {
        for (auto idx : shape.mesh.indices) {
            model->vertices.push_back(*((glm::vec3*)attrib.vertices.data() + idx.vertex_index));
            model->normals.push_back(*((glm::vec3*)attrib.normals.data() + idx.normal_index));
            model->texcoords.push_back(hasTexcoord ?
                *((glm::vec2*)attrib.texcoords.data() + idx.texcoord_index) :
                glm::vec2(0.f)
            );
        }
    }

    std::cout << "\t\t[Vertex count = " << model->vertices.size() << "]" << std::endl;
    meshDataPool[filename] = model;
    return model;
}

MeshData* Resource::loadGLTFMesh(const std::string& filename) {
    auto find = meshDataPool.find(filename);
    if (find != meshDataPool.end()) {
        return find->second;
    }
    auto model = new MeshData;
}

MeshData* Resource::loadModelMeshData(const std::string& filename) {
    if (filename.find(".obj") != filename.npos) {
        return loadOBJMesh(filename);
    }
    else {
        return loadGLTFMesh(filename);
    }
}

Image* Resource::loadTexture(const std::string& filename) {
    auto find = texturePool.find(filename);
    if (find != texturePool.end()) {
        return find->second;
    }
    auto texture = new Image(filename);
    texturePool[filename] = texture;
    return texture;
}

void Resource::clear() {
    for (auto i : meshDataPool) {
        delete i.second;
    }
    meshDataPool.clear();

    for (auto i : texturePool) {
        delete i.second;
    }
    texturePool.clear();
}

Scene::Scene(const std::string& filename) {
    stbi_ldr_to_hdr_gamma(1.f);
    stbi_set_flip_vertically_on_load(true);

    std::cout << "[Scene loading " << filename << " ...]" << std::endl;
    std::cout << " " << std::endl;
    char* fname = (char*)filename.c_str();
    fpIn.open(fname);
    if (!fpIn.is_open()) {
        std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }
    while (fpIn.good()) {
        std::string line;
        utilityCore::safeGetline(fpIn, line);
        if (!line.empty()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            if (tokens[0] == "Material") {
                loadMaterial(tokens[1]);
            }
            else if (tokens[0] == "Object") {
                loadModel(tokens[1]);
            }
            else if (tokens[0] == "Camera") {
                loadCamera();
            }
            else if (tokens[0] == "EnvMap") {
                if (tokens[1] != "Null") {
                    stbi_set_flip_vertically_on_load(false);
                    envMapTexId = addTexture(tokens[1]);
                    stbi_set_flip_vertically_on_load(true);
                }
            }
        }
    }
}

Scene::~Scene() {
}

void Scene::createLightSampler() {
    if (envMapTexId != NullTextureId) {
        auto envMap = textures[envMapTexId];
        std::vector<float> pdf(envMap->width() * envMap->height());

        for (int i = 0; i < envMap->height(); i++) {
            for (int j = 0; j < envMap->width(); j++) {
                int idx = i * envMap->width() + j;
                pdf[idx] = Math::luminance(envMap->data()[idx]) * glm::sin((.5f + i) / envMap->height() * Pi);
            }
        }
        envMapSampler = DiscreteSampler1D<float>(pdf);
        std::cout << "\t[Environment sampler width = " << envMap->width() << ", height = " << envMap->height() <<
            ", sumPower = " << envMapSampler.sumAll << "]\n" << std::endl;

        lightPower.push_back(envMapSampler.sumAll);
    }

    lightSampler = DiscreteSampler1D<float>(lightPower);
    std::cout << "[Light sampler size = " << lightPower.size() << ", sumPower = " <<
        lightSampler.sumAll << "]\n" << std::endl;
}

void Scene::buildDevData() {
    int primId = 0;
    for (const auto& inst : modelInstances) {
        const auto& material = materials[inst.materialId];
        glm::vec3 radianceUnitArea = material.baseColor;
        float powerUnitArea = Math::luminance(radianceUnitArea);

        for (size_t i = 0; i < inst.meshData->vertices.size(); i++) {
            meshData.vertices.push_back(glm::vec3(inst.transform * glm::vec4(inst.meshData->vertices[i], 1.f)));
            meshData.normals.push_back(glm::normalize(inst.normalMat * inst.meshData->normals[i]));
            meshData.texcoords.push_back(inst.meshData->texcoords[i]);

            if (i % 3 == 0) {
                materialIds.push_back(inst.materialId);
            }
            else if (i % 3 == 2) {
                if (material.type == Material::Light) {
                    glm::vec3 v0 = meshData.vertices[i - 2];
                    glm::vec3 v1 = meshData.vertices[i - 1];
                    glm::vec3 v2 = meshData.vertices[i - 0];
                    float area = Math::triangleArea(v0, v1, v2);
                    float power = powerUnitArea * area;

                    lightPrimIds.push_back(primId);
                    lightUnitRadiance.push_back(radianceUnitArea);
                    lightPower.push_back(power);
                    numLightPrims++;
                }
                primId++;
            }
        }
    }

    if (primId == 0) {
        std::cout << "[No mesh data loaded, quit]" << std::endl;
        exit(-1);
    }

    createLightSampler();

    BVHSize = BVHBuilder::build(meshData.vertices, boundingBoxes, BVHNodes);

    hstScene.create(*this);
    devScene = cudaMalloc<DevScene>(1);
    cudaMemcpyHostToDev(devScene, &hstScene, sizeof(DevScene));
    checkCUDAError("Dev Scene");

    meshData.clear();
    boundingBoxes.clear();
    BVHNodes.clear();

    lightPrimIds.clear();
    lightPower.clear();

    lightSampler.clear();
    envMapSampler.clear();
}

void Scene::clear() {
    hstScene.destroy();
    cudaSafeFree(devScene);
}

void Scene::loadModel(const std::string& objId) {
    std::cout << "\t[Object " << objId << "]" << std::endl;

    ModelInstance instance;

    std::string line;
    utilityCore::safeGetline(fpIn, line);

    std::string filename = line;
    std::cout << "\t\t[File " << filename << "]" << std::endl;
    instance.meshData = Resource::loadModelMeshData(filename);

    if (!instance.meshData) {
        std::cout << "\t\t[Fail to load, skipped]" << std::endl;
        while (!line.empty() && fpIn.good()) {
            utilityCore::safeGetline(fpIn, line);
        }
        return;
    }
    
    //link material
    utilityCore::safeGetline(fpIn, line);
    if (!line.empty() && fpIn.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens[1] == "Null") {
            // Null material, create new one
            instance.materialId = addMaterial(Material());
        }
        else {
            if (materialMap.find(tokens[1]) == materialMap.end()) {
                std::cout << "\t\t[Material " << tokens[1] << " doesn't exist]" << std::endl;
                throw;
            }
            instance.materialId = materialMap[tokens[1]];
            std::cout << "\t\t[Link to Material " << tokens[1] << "{" << instance.materialId << "} ...]" << std::endl;
        }
    }

    //load transformations
    utilityCore::safeGetline(fpIn, line);
    while (!line.empty() && fpIn.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (tokens[0] == "Translate") {
            instance.translation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if (tokens[0] == "Rotate") {
            instance.rotation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if (tokens[0] == "Scale") {
            instance.scale = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }

        utilityCore::safeGetline(fpIn, line);
    }

    instance.transform = Math::buildTransformationMatrix(
        instance.translation, instance.rotation, instance.scale
    );
    instance.transfInv = glm::inverse(instance.transform);
    instance.normalMat = glm::transpose(glm::mat3(instance.transfInv));

    modelInstances.push_back(instance);
}

void Scene::loadCamera() {
    std::cout << "\t[Camera]" << std::endl;
    float fovy;

    //load static properties
    for (int i = 0; i < 8; i++) {
        std::string line;
        utilityCore::safeGetline(fpIn, line);
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "Resolution") {
            camera.resolution.x = std::stoi(tokens[1]);
            camera.resolution.y = std::stoi(tokens[2]);
            std::cout << "\t\t[Resolution x = " << camera.resolution.x << ", y = " <<
                camera.resolution.y << "]" << std::endl;
        }
        else if (tokens[0] == "FovY") {
            fovy = std::stof(tokens[1]);
            std::cout << "\t\t[FOV = " << fovy << "]" << std::endl;
        }
        else if (tokens[0] == "LensRadius") {
            camera.lensRadius = std::stof(tokens[1]);
        }
        else if (tokens[0] == "FocalDist") {
            camera.focalDist = std::stof(tokens[1]);
        }
        else if (tokens[0] == "ApertureMask") {
            if (tokens[1] != "Null") {
                std::cout << "\t\t[Aperture skipped]" << std::endl;
            }
        }
        else if (tokens[0] == "Sample") {
            state.iterations = std::stoi(tokens[1]);
        }
        else if (tokens[0] == "Depth") {
            Settings::traceDepth = std::stoi(tokens[1]);
        }
        else if (tokens[0] == "File") {
            state.imageName = tokens[1];
        }
    }

    std::string line;
    utilityCore::safeGetline(fpIn, line);
    while (!line.empty() && fpIn.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "Eye") {
            camera.position = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        } else if (tokens[0] == "Rotation") {
            camera.rotation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        } else if (tokens[0] == "Up") {
            camera.up = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        utilityCore::safeGetline(fpIn, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (Pi / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / Pi;
    camera.fov = glm::vec2(fovx, fovy);
    camera.tanFovY = glm::tan(glm::radians(fovy * 0.5f));
    camera.update();

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

int Scene::addMaterial(const Material& material) {
    materials.push_back(material);
    return materials.size() - 1;
}

int Scene::addTexture(const std::string& filename) {
    auto texture = Resource::loadTexture(filename);
    auto find = textureMap.find(texture);
    if (find != textureMap.end()) {
        return find->second;
    }
    else {
        int size = textureMap.size();
        textureMap[texture] = size;
        textures.push_back(texture);
        return size;
    }
}

void Scene::loadMaterial(const std::string& matId) {
    std::cout << "\t[Material " << matId << "]" << std::endl;
    Material material;

    //load static properties
    for (int i = 0; i < 6; i++) {
        std::string line;
        utilityCore::safeGetline(fpIn, line);
        auto tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "Type") {
            material.type = MaterialTypeTokenMap[tokens[1]];
            std::cout << "\t\t[Type " << tokens[1] << "]" << std::endl;
        }
        else if (tokens[0] == "BaseColor") {
            if (tokens.size() > 2) {
                glm::vec3 baseColor(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
                material.baseColor = baseColor;
            }
            else if (tokens[1] == "Procedural") {
                material.baseColorMapId = ProceduralTexId;
                std::cout << "\t\t[BaseColor use procedural texture]" << std::endl;
            }
            else {
                material.baseColorMapId = addTexture(tokens[1]);
                std::cout << "\t\t[BaseColor use texture " << tokens[1] << "]" << std::endl;
            }
        }
        else if (tokens[0] == "Metallic") {
            if (std::isdigit(tokens[1][tokens[1].length() - 1])) {
                material.metallic = std::stof(tokens[1]);
            }
            else {
                material.metallicMapId = addTexture(tokens[1]);
                std::cout << "\t\t[Metallic use texture " << tokens[1] << "]" << std::endl;
            }
        }
        else if (tokens[0] == "Roughness") {
            if (std::isdigit(tokens[1][tokens[1].length() - 1])) {
                material.roughness = std::stof(tokens[1]);
            }
            else {
                material.roughnessMapId = addTexture(tokens[1]);
                std::cout << "\t\t[Roughness use texture " << tokens[1] << "]" << std::endl;
            }
        }
        else if (tokens[0] == "Ior") {
            material.ior = std::stof(tokens[1]);
        }
        else if (tokens[0] == "NormalMap") {
            if (tokens[1] != "Null") {
                material.normalMapId = addTexture(tokens[1]);
                std::cout << "\t\t[NormalMap use texture " << tokens[1] << "]" << std::endl;
            }
        }
    }
    materialMap[matId] = materials.size();
    materials.push_back(material);
}

void DevScene::create(const Scene& scene) {
    // Put all texture devData in a big buffer
    // and setup device texture objects to manage
    std::vector<DevTextureObj> textureObjs;

    size_t textureTotalSize = 0;
    for (auto tex : scene.textures) {
        textureTotalSize += tex->byteSize();
    }
    cudaMalloc(&textureData, textureTotalSize);
    checkCUDAError("DevScene::texture");

    int textureOffset = 0;
    for (auto tex : scene.textures) {
        cudaMemcpyHostToDev(textureData + textureOffset, tex->data(), tex->byteSize());
        checkCUDAError("DevScene::texture::copy");
        textureObjs.push_back({ tex, textureData + textureOffset });
        textureOffset += tex->byteSize() / sizeof(glm::vec3);
    }
    textures = cudaMalloc<DevTextureObj>(textureObjs.size());
    checkCUDAError("DevScene::textureObjs::malloc");
    cudaMemcpyHostToDev(textures, textureObjs.data(), textureObjs.size() * sizeof(DevTextureObj));
    checkCUDAError("DevScene::textureObjs::copy");

    cudaMalloc(&materials, byteSizeOf(scene.materials));
    cudaMemcpyHostToDev(materials, scene.materials.data(), byteSizeOf(scene.materials));

    cudaMalloc(&materialIds, byteSizeOf(scene.materialIds));
    cudaMemcpyHostToDev(materialIds, scene.materialIds.data(), byteSizeOf(scene.materialIds));
    checkCUDAError("DevScene::material");

    cudaMalloc(&vertices, byteSizeOf(scene.meshData.vertices));
    cudaMemcpyHostToDev(vertices, scene.meshData.vertices.data(), byteSizeOf(scene.meshData.vertices));

    cudaMalloc(&normals, byteSizeOf(scene.meshData.normals));
    cudaMemcpyHostToDev(normals, scene.meshData.normals.data(), byteSizeOf(scene.meshData.normals));

    cudaMalloc(&texcoords, byteSizeOf(scene.meshData.texcoords));
    cudaMemcpyHostToDev(texcoords, scene.meshData.texcoords.data(), byteSizeOf(scene.meshData.texcoords));

    cudaMalloc(&boundingBoxes, byteSizeOf(scene.boundingBoxes));
    cudaMemcpyHostToDev(boundingBoxes, scene.boundingBoxes.data(), byteSizeOf(scene.boundingBoxes));

    for (int i = 0; i < 6; i++) {
        cudaMalloc(&BVHNodes[i], byteSizeOf(scene.BVHNodes[i]));
        cudaMemcpyHostToDev(BVHNodes[i], scene.BVHNodes[i].data(), byteSizeOf(scene.BVHNodes[i]));
    }
    BVHSize = scene.BVHSize;

    cudaMalloc(&lightPrimIds, byteSizeOf(scene.lightPrimIds));
    cudaMemcpyHostToDev(lightPrimIds, scene.lightPrimIds.data(), byteSizeOf(scene.lightPrimIds));

    cudaMalloc(&lightUnitRadiance, byteSizeOf(scene.lightUnitRadiance));
    cudaMemcpyHostToDev(lightUnitRadiance, scene.lightUnitRadiance.data(), byteSizeOf(scene.lightUnitRadiance));

    checkCUDAError("DevScene::meshData");

    lightSampler.create(scene.lightSampler);
    sumLightPowerInv = 1.f / scene.lightSampler.sumAll;

    if (scene.envMapTexId != NullTextureId) {
        envMap = textures + scene.envMapTexId;
        envMapSampler.create(scene.envMapSampler);
    }

#if SAMPLER_USE_SOBOL
    std::ifstream sobolFile("sobol_10k_200.bin", std::ios::in | std::ios::binary);
    std::vector<char> sobolData(SobolSampleNum * SobolSampleDim * sizeof(uint32_t));
    sobolFile.read(sobolData.data(), byteSizeOf(sobolData));
    cudaMalloc(&sampleSequence, byteSizeOf(sobolData));
    cudaMemcpyHostToDev(sampleSequence, sobolData.data(), byteSizeOf(sobolData));
#endif

    checkCUDAError("DevScene::samplers");
}

void DevScene::destroy() {
    cudaSafeFree(textureData);
    cudaSafeFree(textures);
    cudaSafeFree(materials);
    cudaSafeFree(materialIds);
    
    cudaSafeFree(vertices);
    cudaSafeFree(normals);
    cudaSafeFree(texcoords);
    cudaSafeFree(boundingBoxes);

    for (int i = 0; i < 6; i++) {
        cudaSafeFree(BVHNodes[i]);
    }

    cudaSafeFree(lightPrimIds);
    cudaSafeFree(lightUnitRadiance);
    lightSampler.destroy();
    envMapSampler.destroy();

    cudaSafeFree(sampleSequence);
}
