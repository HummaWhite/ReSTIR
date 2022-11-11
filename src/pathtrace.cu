#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "material.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "mathUtil.h"
#include "sampler.h"

#define PixelIdxForTerminated -1

static Scene* hstScene = nullptr;
static GuiDataContainer* guiData = nullptr;
static PathSegment* devPaths = nullptr;
static PathSegment* devTerminatedPaths = nullptr;
static Intersection* devIntersections = nullptr;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static thrust::device_ptr<PathSegment> devPathsThr;
static thrust::device_ptr<PathSegment> devTerminatedPathsThr;
static thrust::device_ptr<Intersection> devIntersectionsThr;

static int looper = 0;
 
void InitDataContainer(GuiDataContainer* imGuiData) {
	guiData = imGuiData;
}

void pathTraceInit(Scene* scene) {
	hstScene = scene;

	const Camera& cam = hstScene->camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	devPaths = cudaMalloc<PathSegment>(pixelcount);
	cudaMalloc(&devTerminatedPaths, pixelcount * sizeof(PathSegment));
	devPathsThr = thrust::device_ptr<PathSegment>(devPaths);
	devTerminatedPathsThr = thrust::device_ptr<PathSegment>(devTerminatedPaths);

	cudaMalloc(&devIntersections, pixelcount * sizeof(Intersection));
	cudaMemset(devIntersections, 0, pixelcount * sizeof(Intersection));
	devIntersectionsThr = thrust::device_ptr<Intersection>(devIntersections);

	checkCUDAError("pathTraceInit");
}

void pathTraceFree() {
	cudaSafeFree(devPaths);
	cudaSafeFree(devTerminatedPaths);
	cudaSafeFree(devIntersections);
#if ENABLE_GBUFFER
	cudaSafeFree(devGBuffer);
#endif
}

__global__ void sendImageToPBO(uchar4* pbo, glm::vec3* image, int width, int height, int toneMapping, float scale) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
	int index = y * width + x;

	// Tonemapping and gamma correction
	glm::vec3 color = image[index] * scale;

	switch (toneMapping) {
	case ToneMapping::Filmic:
		color = Math::filmic(color);
		break;
	case ToneMapping::ACES:
		color = Math::ACES(color);
		break;
	case ToneMapping::None:
		break;
	}
	color = Math::correctGamma(color);

	glm::ivec3 iColor = glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
	pbo[index] = make_uchar4(iColor.x, iColor.y, iColor.z, 0);
}

__global__ void sendImageToPBO(uchar4* pbo, glm::vec2* image, int width, int height) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
	int index = y * width + x;

	glm::vec3 color = glm::vec3(image[index], 0.f);
	color = Math::correctGamma(color);

	glm::ivec3 iColor = glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
	pbo[index] = make_uchar4(iColor.x, iColor.y, iColor.z, 0);
}

__global__ void sendImageToPBO(uchar4* pbo, float* image, int width, int height) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
	int index = y * width + x;

	glm::vec3 color = glm::vec3(image[index]);
	color = Math::correctGamma(color);

	glm::ivec3 iColor = glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
	pbo[index] = make_uchar4(iColor.x, iColor.y, iColor.z, 0);
}

__global__ void sendImageToPBO(uchar4* pbo, int* image, int width, int height) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
	int index = y * width + x;
	int px = image[index] % width;
	int py = image[index] / height;

	glm::vec3 color = glm::vec3(glm::vec2(px, py) / glm::vec2(width, height), 0.f);
	color = Math::correctGamma(color);

	glm::ivec3 iColor = glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));
	pbo[index] = make_uchar4(iColor.x, iColor.y, iColor.z, 0);
}

void copyImageToPBO(uchar4* devPBO, glm::vec3* devImage, int width, int height, int toneMapping, float scale) {
	const int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height, toneMapping, scale);
}

void copyImageToPBO(uchar4* devPBO, glm::vec2* devImage, int width, int height) {
	const int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height);
}

void copyImageToPBO(uchar4* devPBO, float* devImage, int width, int height) {
	const int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height);
}

void copyImageToPBO(uchar4* devPBO, int* devImage, int width, int height) {
	const int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	sendImageToPBO<<<blockNum, blockSize>>>(devPBO, devImage, width, height);
}

__global__ void generateRayFromCamera(
	DevScene* scene, Camera cam, 
	int iter, int traceDepth, PathSegment* pathSegments
) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];
		Sampler rng = makeSeededRandomEngine(iter, index, traceDepth, scene->sampleSequence);

		segment.ray = cam.sample(x, y, sample4D(rng));
		segment.throughput = glm::vec3(1.f);
		segment.directIllum = glm::vec3(0.f);
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void previewGBuffer(int iter, DevScene* scene, Camera cam, glm::vec3* image, int kind) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	int index = y * cam.resolution.x + x;
	Sampler rng = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);

	Ray ray = cam.sample(x, y, sample4D(rng));
	Intersection intersec;
	scene->intersect(ray, intersec);

	if (kind == 0) {
		image[index] += intersec.pos;
	}
	else if (kind == 1) {
		if (intersec.primId != NullPrimitive) {
			Material m = scene->getTexturedMaterialAndSurface(intersec);
		}
		image[index] += (intersec.norm + 1.f) * .5f;
	}
	else if (kind == 2) {
		image[index] += glm::vec3(intersec.uv, 1.f);
	}
}

__global__ void computeIntersections(
	int depth,
	int numPaths,
	PathSegment* pathSegments,
	DevScene* scene,
	Intersection* intersections
) {
	int pathIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pathIdx >= numPaths) {
		return;
	}

	Intersection intersec;
	PathSegment segment = pathSegments[pathIdx];

	scene->intersect(segment.ray, intersec);

	if (intersec.primId != NullPrimitive) {
		if (scene->materials[intersec.matId].type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
			if (glm::dot(intersec.norm, segment.ray.direction) < 0.f) {
				intersec.primId = NullPrimitive;
			}
			else
#endif
			if (depth != 0) {
				// If not first ray, preserve previous sampling information for
				// MIS calculation
				intersec.prevPos = segment.ray.origin;
			}
		}
		else {
			intersec.wo = -segment.ray.direction;
		}
	}
	intersections[pathIdx] = intersec;
}

__global__ void pathIntegSampleSurface(
	int looper, int iter,
	int depth,
	PathSegment* segments,
	Intersection* intersections,
	DevScene* scene,
	int numPaths
) {
	const int SamplesOneIter = 7;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numPaths) {
		return;
	}
	Intersection intersec = intersections[idx];
	PathSegment& segment = segments[idx];

	if (intersec.primId == NullPrimitive) {
		if (scene->envMap != nullptr) {
			if (scene->envMap != nullptr) {
				glm::vec3 w = segment.ray.direction;
				glm::vec3 radiance = scene->envMap->linearSample(Math::toPlane(w)) * segment.throughput;

				if (depth == 0) {
					segment.directIllum += radiance * segment.throughput;
				}
				else {
					float weight = segment.prev.deltaSample ? 1.f :
						Math::powerHeuristic(segment.prev.BSDFPdf, scene->environmentMapPdf(w));
					segment.directIllum += radiance * weight;
				}
			}
		}
		segment.remainingBounces = 0;

		if (Math::luminance(segment.directIllum) < 1e-4f) {
			segment.pixelIndex = PixelIdxForTerminated;
		}
		return;
	}

	Sampler rng = makeSeededRandomEngine(looper, idx, 4 + depth * SamplesOneIter, scene->sampleSequence);

	Material material = scene->getTexturedMaterialAndSurface(intersec);

	glm::vec3 accRadiance(0.f);

	if (material.type == Material::Type::Light) {
		PrevBSDFSampleInfo prev = segment.prev;

		glm::vec3 radiance = material.baseColor;
		if (depth == 0) {
			accRadiance += radiance;
		}
		else if (prev.deltaSample) {
			accRadiance += radiance * segment.throughput;
		}
		else {
			float lightPdf = Math::pdfAreaToSolidAngle(Math::luminance(radiance) * scene->sumLightPowerInv *
				scene->getPrimitiveArea(intersec.primId), intersec.prevPos, intersec.pos, intersec.norm);
			float BSDFPdf = prev.BSDFPdf;
			accRadiance += radiance * segment.throughput * Math::powerHeuristic(BSDFPdf, lightPdf);
		}
		segment.remainingBounces = 0;
	}
	else {
		bool deltaBSDF = (material.type == Material::Type::Dielectric);
		if (material.type != Material::Type::Dielectric && glm::dot(intersec.norm, intersec.wo) < 0.f) {
			intersec.norm = -intersec.norm;
		}

		if (!deltaBSDF) {
			glm::vec3 radiance;
			glm::vec3 wi;
			float lightPdf = scene->sampleDirectLight(intersec.pos, sample4D(rng), radiance, wi);

			if (lightPdf > 0.f) {
				float BSDFPdf = material.pdf(intersec.norm, intersec.wo, wi);
				accRadiance += segment.throughput * material.BSDF(intersec.norm, intersec.wo, wi) *
					radiance * Math::satDot(intersec.norm, wi) / lightPdf * Math::powerHeuristic(lightPdf, BSDFPdf);
			}
		}

		BSDFSample sample;
		material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);

		if (sample.type == BSDFSampleType::Invalid) {
			// Terminate path if sampling fails
			segment.remainingBounces = 0;
		}
		else if (sample.pdf < 1e-8f) {
			segment.remainingBounces = 0;
		}
		else {
			bool deltaSample = (sample.type & BSDFSampleType::Specular);
			segment.throughput *= sample.bsdf / sample.pdf *
				(deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));
			segment.ray = makeOffsetedRay(intersec.pos, sample.dir);
			segment.prev = { sample.pdf, deltaSample };
			segment.remainingBounces--;
		}
	}
	segment.directIllum += accRadiance;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths) {
		PathSegment iterationPath = iterationPaths[index];
		if (iterationPath.pixelIndex >= 0 && iterationPath.remainingBounces <= 0) {
			glm::vec3 r = iterationPath.directIllum;
			if (isnan(r.x) || isnan(r.y) || isnan(r.z) || isinf(r.x) || isinf(r.y) || isinf(r.z)) {
				return;
			}
			image[iterationPath.pixelIndex] += glm::clamp(r, glm::vec3(0.f), glm::vec3(FLT_MAX / 10.f));
		}
	}
}

__global__ void singleKernelPT(
	int looper, int iter, int maxDepth,
	DevScene* scene, Camera cam,
	glm::vec3* directIllum, glm::vec3* indirectIllum
) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	glm::vec3 direct(0.f);
	glm::vec3 indirect(0.f);

	int index = y * cam.resolution.x + x;
	Sampler rng = makeSeededRandomEngine(looper, index, 0, scene->sampleSequence);

	Ray ray = cam.sample(x, y, sample4D(rng));
	Intersection intersec;
	scene->intersect(ray, intersec);

	if (intersec.primId == NullPrimitive) {
		direct = glm::vec3(1.f);
		goto WriteRadiance;
	}

	Material material = scene->getTexturedMaterialAndSurface(intersec);
#if DENOISER_DEMODULATE
	glm::vec3 albedo = material.baseColor;
	material.baseColor = glm::vec3(1.f);
#endif

	if (material.type == Material::Type::Light) {
		direct = glm::vec3(1.f);
		goto WriteRadiance;
	}

	glm::vec3 throughput(1.f);
	intersec.wo = -ray.direction;

	for (int depth = 1; depth <= maxDepth; depth++) {
		bool deltaBSDF = (material.type == Material::Type::Dielectric);

		if (material.type != Material::Type::Dielectric && glm::dot(intersec.norm, intersec.wo) < 0.f) {
			intersec.norm = -intersec.norm;
		}

		if (!deltaBSDF) {
			glm::vec3 radiance;
			glm::vec3 wi;
			float lightPdf = scene->sampleDirectLight(intersec.pos, sample4D(rng), radiance, wi);

			if (lightPdf > 0.f) {
				float BSDFPdf = material.pdf(intersec.norm, intersec.wo, wi);
				(depth == 1 ? direct : indirect) += throughput * material.BSDF(intersec.norm, intersec.wo, wi) *
					radiance * Math::satDot(intersec.norm, wi) / lightPdf * Math::powerHeuristic(lightPdf, BSDFPdf);
			}
		}

		BSDFSample sample;
		material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);

		if (sample.type == BSDFSampleType::Invalid) {
			// Terminate path if sampling fails
			break;
		}
		else if (sample.pdf < 1e-8f) {
			break;
		}

		bool deltaSample = (sample.type & BSDFSampleType::Specular);
		throughput *= sample.bsdf / sample.pdf *
			(deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));
		
		ray = makeOffsetedRay(intersec.pos, sample.dir);

		glm::vec3 curPos = intersec.pos;
		scene->intersect(ray, intersec);
		intersec.wo = -ray.direction;

		if (intersec.primId == NullPrimitive) {
			if (scene->envMap != nullptr) {
				glm::vec3 radiance = scene->envMap->linearSample(Math::toPlane(ray.direction))
					* throughput;

				float weight = deltaSample ? 1.f :
					Math::powerHeuristic(sample.pdf, scene->environmentMapPdf(ray.direction));
				indirect += radiance * weight;
			}
			break;
		}
		material = scene->getTexturedMaterialAndSurface(intersec);

		if (material.type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
			if (glm::dot(intersec.norm, ray.direction) < 0.f) {
				break;
			}
#endif
			glm::vec3 radiance = material.baseColor;

			float weight = deltaSample ? 1.f : Math::powerHeuristic(
				sample.pdf,
				Math::pdfAreaToSolidAngle(Math::luminance(radiance) * scene->sumLightPowerInv *
					scene->getPrimitiveArea(intersec.primId), curPos, intersec.pos, intersec.norm)
			);
			indirect += radiance * throughput * weight;
			break;
		}
	}
WriteRadiance:
	if (Math::hasNanOrInf(direct)) {
		direct = glm::vec3(0.f);
	}
	if (Math::hasNanOrInf(indirect)) {
		indirect = glm::vec3(0.f);
	}

	direct = Math::HDRToLDR(direct);
	indirect = Math::HDRToLDR(indirect);
	directIllum[index] = (directIllum[index] * float(iter) + direct) / float(iter + 1);
	indirectIllum[index] = (indirectIllum[index] * float(iter) + indirect) / float(iter + 1);
}

__global__ void BVHVisualize(int iter, DevScene* scene, Camera cam, glm::vec3* image) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	int index = y * cam.resolution.x + x;

	Sampler rng = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);
	Ray ray = cam.sample(x, y, sample4D(rng));

	Intersection intersec;
	scene->visualizedIntersect(ray, intersec);

	float logDepth = 0.f;
	int size = scene->BVHSize;
	while (size) {
		logDepth += 1.f;
		size >>= 1;
	}
	image[index] += glm::vec3(float(intersec.primId) / logDepth * .06f);
}

struct CompactTerminatedPaths {
	__host__ __device__ bool operator() (const PathSegment& segment) {
		return !(segment.pixelIndex >= 0 && segment.remainingBounces <= 0);
	}
};

struct RemoveInvalidPaths {
	__host__ __device__ bool operator() (const PathSegment& segment) {
		return segment.pixelIndex < 0 || segment.remainingBounces <= 0;
	}
};

void pathTrace(glm::vec3* devDirectIllum, glm::vec3* devIndirectIllum, int iter) {
	const Camera& cam = hstScene->camera;

	const int BlockSizeSinglePTX = 8;
	const int BlockSizeSinglePTY = 8;
	int blockNumSinglePTX = ceilDiv(cam.resolution.x, BlockSizeSinglePTX);
	int blockNumSinglePTY = ceilDiv(cam.resolution.y, BlockSizeSinglePTY);

	dim3 singlePTBlockNum(blockNumSinglePTX, blockNumSinglePTY);
	dim3 singlePTBlockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);

	singleKernelPT<<<singlePTBlockNum, singlePTBlockSize>>>(
		looper, iter, Settings::traceDepth, hstScene->devScene, cam, devDirectIllum, devIndirectIllum
	);

	checkCUDAError("pathTrace");
	looper = (looper + 1) % SobolSampleNum;
}