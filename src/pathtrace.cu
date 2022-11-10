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

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* Image, int toneMapping) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);

		// Tonemapping and gamma correction
		glm::vec3 color = Image[index] / float(iter);

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

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = iColor.x;
		pbo[index].y = iColor.y;
		pbo[index].z = iColor.z;
	}
}

#define PixelIdxForTerminated -1

static Scene* hstScene = nullptr;
static GuiDataContainer* guiData = nullptr;
static glm::vec3* devImage = nullptr;
static PathSegment* devPaths = nullptr;
static PathSegment* devTerminatedPaths = nullptr;
static Intersection* devIntersections = nullptr;
static int* devIntersecMatKeys = nullptr;
static int* devSegmentMatKeys = nullptr;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static thrust::device_ptr<PathSegment> devPathsThr;
static thrust::device_ptr<PathSegment> devTerminatedPathsThr;

static thrust::device_ptr<Intersection> devIntersectionsThr;
static thrust::device_ptr<int> devIntersecMatKeysThr;
static thrust::device_ptr<int> devSegmentMatKeysThr;

static glm::vec3* devGBufferPos = nullptr;
static glm::vec3* devGBufferNorm = nullptr;

#if ENABLE_GBUFFER
static Intersection* devGBuffer = nullptr;
#endif
 
void InitDataContainer(GuiDataContainer* imGuiData) {
	guiData = imGuiData;
}

#if ENABLE_GBUFFER
#endif

__global__ void renderGBuffer(DevScene* scene, Camera cam, Intersection *GBuffer) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx >= cam.resolution.x || idy >= cam.resolution.y) {
		return;
	}

	float aspect = float(cam.resolution.x) / cam.resolution.y;
	float tanFovY = glm::tan(glm::radians(cam.fov.y));
	glm::vec2 pixelSize = 1.f / glm::vec2(cam.resolution);
	glm::vec2 scr = glm::vec2(idx, idy) * pixelSize;
	glm::vec2 ruv = scr + pixelSize * glm::vec2(.5f);
	ruv = 1.f - ruv * 2.f;

	glm::vec3 pLens(0.f);
	glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
	glm::vec3 dir = pFocusPlane - pLens;

	Ray ray;
	ray.direction = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);
	ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;

	Intersection intersec;
	scene->intersect(ray, intersec);

	if (intersec.primId != NullPrimitive) {
		if (scene->materials[intersec.matId].type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
			if (glm::dot(intersec.norm, ray.direction) < 0.f) {
				intersec.primId = NullPrimitive;
			}
#endif
		}
		else {
			intersec.wo = -ray.direction;
		}
	}
	GBuffer[idy * cam.resolution.x + idx] = intersec;
}

void pathTraceInit(Scene* scene) {
	hstScene = scene;

	const Camera& cam = hstScene->camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&devImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(devImage, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&devPaths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&devTerminatedPaths, pixelcount * sizeof(PathSegment));
	devPathsThr = thrust::device_ptr<PathSegment>(devPaths);
	devTerminatedPathsThr = thrust::device_ptr<PathSegment>(devTerminatedPaths);

	cudaMalloc(&devIntersections, pixelcount * sizeof(Intersection));
	cudaMemset(devIntersections, 0, pixelcount * sizeof(Intersection));
	devIntersectionsThr = thrust::device_ptr<Intersection>(devIntersections);

	cudaMalloc(&devIntersecMatKeys, pixelcount * sizeof(int));
	cudaMalloc(&devSegmentMatKeys, pixelcount * sizeof(int));
	devIntersecMatKeysThr = thrust::device_ptr<int>(devIntersecMatKeys);
	devSegmentMatKeysThr = thrust::device_ptr<int>(devSegmentMatKeys);
	checkCUDAError("pathTraceInit");

#if ENABLE_GBUFFER
	cudaMalloc(&devGBuffer, pixelcount * sizeof(Intersection));
	const int BlockSize = 8;
	dim3 blockSize(BlockSize, BlockSize);

	dim3 blockNum((cam.resolution.x + BlockSize - 1) / BlockSize,
		(cam.resolution.y + BlockSize - 1) / BlockSize
	);
	renderGBuffer<<<blockNum, blockSize>>>(hstScene->devScene, cam, devGBuffer);
	checkCUDAError("GBuffer");
	std::cout << "[GBuffer generated]" << std::endl;
#endif
}

void pathTraceFree() {
	cudaSafeFree(devImage);  // no-op if devImage is null
	cudaSafeFree(devPaths);
	cudaSafeFree(devTerminatedPaths);
	cudaSafeFree(devIntersections);
	cudaSafeFree(devIntersecMatKeys);
	cudaSafeFree(devSegmentMatKeys);
#if ENABLE_GBUFFER
	cudaSafeFree(devGBuffer);
#endif
}

/**
 * Antialiasing and physically based camera (lens effect)
 */
__device__ Ray sampleCamera(DevScene* scene, const Camera& cam, int x, int y, glm::vec4 r) {
	Ray ray;
#if CAMERA_PANORAMA
	float u = (x - .5f + r.x) / cam.resolution.x - .5f;
	float v = (y - .5f + r.y) / cam.resolution.y;
	glm::vec3 dir = Math::toSphere(glm::vec2(u, v));
	dir = cam.right * dir.x + cam.up * dir.y + cam.view * dir.z;
	ray.direction = dir;
	ray.origin = cam.position;
#else
	float aspect = float(cam.resolution.x) / cam.resolution.y;
	float tanFovY = glm::tan(glm::radians(cam.fov.y));
	glm::vec2 pixelSize = 1.f / glm::vec2(cam.resolution);
	glm::vec2 scr = glm::vec2(x, y) * pixelSize;
	glm::vec2 ruv = scr + pixelSize * glm::vec2(r.x, r.y);
	ruv = 1.f - ruv * 2.f;

	glm::vec2 pAperture;
	if (scene->apertureMask != nullptr) {
		int id = scene->apertureSampler.sample(r.z, r.w);
		pAperture.x = glm::fract((id + .5f) / scene->apertureMask->width);
		pAperture.y = (id / scene->apertureMask->width + .5f) / scene->apertureMask->height;
		pAperture = pAperture * 2.f - 1.f;
	}
	else {
		pAperture = Math::toConcentricDisk(r.z, r.w);
	}

	glm::vec3 pLens = glm::vec3(pAperture * cam.lensRadius, 0.f);

	glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
	glm::vec3 dir = pFocusPlane - pLens;
	ray.direction = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);
	ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;
#endif
	return ray;
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

		segment.ray = sampleCamera(scene, cam, x, y, sample4D(rng));
		segment.throughput = glm::vec3(1.f);
		segment.radiance = glm::vec3(0.f);
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

	Ray ray = sampleCamera(scene, cam, x, y, sample4D(rng));
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
	Intersection* intersections,
	int* materialKeys,
	bool sortMaterial
#if ENABLE_GBUFFER
	, Intersection* GBuffer
#endif
) {
	int pathIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pathIdx >= numPaths) {
		return;
	}

	Intersection intersec;
	PathSegment segment = pathSegments[pathIdx];
#if ENABLE_GBUFFER
	if (depth == 0) {
		intersections[pathIdx] = GBuffer[pathIdx];
		return;
	}
#endif

#if BVH_DISABLE
	scene->naiveIntersect(segment.ray, intersec);
#else
	scene->intersect(segment.ray, intersec);
#endif

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
				if (sortMaterial) {
					intersec.prev = segment.prev;
				}
			}
		}
		else {
			intersec.wo = -segment.ray.direction;
		}
		if (sortMaterial) {
			materialKeys[pathIdx] = intersec.matId;
		}
	}
	else if (sortMaterial) {
		materialKeys[pathIdx] = -1;
	}
	intersections[pathIdx] = intersec;
}

__global__ void pathIntegSampleSurface(
	int iter,
	int depth,
	PathSegment* segments,
	Intersection* intersections,
	DevScene* scene,
	int numPaths,
	bool sortMaterial
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
					segment.radiance += radiance * segment.throughput;
				}
				else {
					float weight = segment.prev.deltaSample ? 1.f :
						Math::powerHeuristic(segment.prev.BSDFPdf, scene->environmentMapPdf(w));
					segment.radiance += radiance * weight;
				}
			}
		}
		segment.remainingBounces = 0;

		if (Math::luminance(segment.radiance) < 1e-4f) {
			segment.pixelIndex = PixelIdxForTerminated;
		}
		return;
	}

	Sampler rng = makeSeededRandomEngine(iter, idx, 4 + depth * SamplesOneIter, scene->sampleSequence);

	Material material = scene->getTexturedMaterialAndSurface(intersec);

	glm::vec3 accRadiance(0.f);

	if (material.type == Material::Type::Light) {
		PrevBSDFSampleInfo prev = sortMaterial ? intersec.prev : segment.prev;

		glm::vec3 radiance = material.baseColor;
		if (depth == 0) {
			accRadiance += radiance;
		}
		else if (prev.deltaSample) {
			accRadiance += radiance * segment.throughput;
		}
		else {
			float lightPdf = Math::pdfAreaToSolidAngle(Math::luminance(radiance) * scene->sumLightPowerInv,
				intersec.prevPos, intersec.pos, intersec.norm);
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
	segment.radiance += accRadiance;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths) {
		PathSegment iterationPath = iterationPaths[index];
		if (iterationPath.pixelIndex >= 0 && iterationPath.remainingBounces <= 0) {
			glm::vec3 r = iterationPath.radiance;
			if (isnan(r.x) || isnan(r.y) || isnan(r.z) || isinf(r.x) || isinf(r.y) || isinf(r.z)) {
				return;
			}
			image[iterationPath.pixelIndex] += glm::clamp(r, glm::vec3(0.f), glm::vec3(FLT_MAX / 10.f));
		}
	}
}

__global__ void singleKernelPT(int iter, int maxDepth, DevScene* scene, Camera cam, glm::vec3* image) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	glm::vec3 accRadiance(0.f);

	int index = y * cam.resolution.x + x;
	Sampler rng = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);

	Ray ray = sampleCamera(scene, cam, x, y, sample4D(rng));
	Intersection intersec;
	scene->intersect(ray, intersec);

	if (intersec.primId == NullPrimitive) {
		if (scene->envMap != nullptr) {
			glm::vec2 uv = Math::toPlane(ray.direction);
			accRadiance += scene->envMap->linearSample(uv);
		}
		goto WriteRadiance;
	}

	Material material = scene->getTexturedMaterialAndSurface(intersec);

	if (material.type == Material::Type::Light) {
		if (glm::dot(intersec.norm, ray.direction) > 0.f) {
			accRadiance = material.baseColor;
		}
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
				accRadiance += throughput * material.BSDF(intersec.norm, intersec.wo, wi) *
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

				accRadiance += radiance * weight;
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
				Math::pdfAreaToSolidAngle(Math::luminance(radiance) * scene->sumLightPowerInv,
					curPos, intersec.pos, intersec.norm)
			);
			accRadiance += radiance * throughput * weight;
			break;
		}
	}
WriteRadiance:
	if (isnan(accRadiance.x) || isnan(accRadiance.y) || isnan(accRadiance.z) ||
		isinf(accRadiance.x) || isinf(accRadiance.y) || isinf(accRadiance.z)) {
		return;
	}
	image[index] += accRadiance;
}

__global__ void BVHVisualize(int iter, DevScene* scene, Camera cam, glm::vec3* image) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	int index = y * cam.resolution.x + x;

	Sampler rng = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);
	Ray ray = sampleCamera(scene, cam, x, y, sample4D(rng));

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

void pathTrace(uchar4* pbo, int frame, int iter) {
	const Camera& cam = hstScene->camera;
	const int pixelCount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2D(8, 8);
	const dim3 blocksPerGrid2D(
		(cam.resolution.x + blockSize2D.x - 1) / blockSize2D.x,
		(cam.resolution.y + blockSize2D.y - 1) / blockSize2D.y);

	int depth = 0;
	int numPaths = pixelCount;

	auto devTerminatedThr = devTerminatedPathsThr;

	if (Settings::tracer == Tracer::Streamed) {
		generateRayFromCamera<<<blocksPerGrid2D, blockSize2D>>>(hstScene->devScene, cam, iter, Settings::traceDepth, devPaths);
		checkCUDAError("PT::generateRayFromCamera");
		cudaDeviceSynchronize();

		bool iterationComplete = false;
		while (!iterationComplete) {
			// clean shading chunks
			cudaMemset(devIntersections, 0, pixelCount * sizeof(Intersection));

			// tracing
			const int BlockSizeIntersec = 128;
			int blockNumIntersec = (numPaths + BlockSizeIntersec - 1) / BlockSizeIntersec;
			computeIntersections<<<blockNumIntersec, BlockSizeIntersec>>>(
				depth, numPaths, devPaths, hstScene->devScene, devIntersections, devIntersecMatKeys, Settings::sortMaterial
#if ENABLE_GBUFFER
				, devGBuffer
#endif
			);
			checkCUDAError("PT::computeInteractions");
			cudaDeviceSynchronize();

			if (Settings::sortMaterial) {
				cudaMemcpyDevToDev(devSegmentMatKeys, devIntersecMatKeys, numPaths * sizeof(int));
				thrust::sort_by_key(devIntersecMatKeysThr, devIntersecMatKeysThr + numPaths, devIntersectionsThr);
				thrust::sort_by_key(devSegmentMatKeysThr, devSegmentMatKeysThr + numPaths, devPathsThr);
			}

			const int BlockSizeSample = 64;
			int blockNumSample = (numPaths + BlockSizeSample - 1) / BlockSizeSample;

			pathIntegSampleSurface<<<blockNumSample, BlockSizeSample>>>(
				iter, depth, devPaths, devIntersections, hstScene->devScene, numPaths, Settings::sortMaterial
			);
			checkCUDAError("PT::sampleSurface");
			cudaDeviceSynchronize();

			// Compact paths that are terminated but carry contribution into a separate buffer
			devTerminatedThr = thrust::remove_copy_if(devPathsThr, devPathsThr + numPaths, devTerminatedThr, CompactTerminatedPaths());
			// Only keep active paths
			auto end = thrust::remove_if(devPathsThr, devPathsThr + numPaths, RemoveInvalidPaths());
			numPaths = end - devPathsThr;
			//std::cout << "Remaining paths: " << numPaths << "\n";

			iterationComplete = (numPaths == 0);
			depth++;

			if (guiData != nullptr) {
				guiData->TracedDepth = depth;
			}
		}

		// Assemble this iteration and apply it to the image
		const int BlockSizeGather = 128;
		dim3 numBlocksPixels = (pixelCount + BlockSizeGather - 1) / BlockSizeGather;
		int numContributing = devTerminatedThr.get() - devTerminatedPaths;
		finalGather<<<numBlocksPixels, BlockSizeGather>>>(numContributing, devImage, devTerminatedPaths);
	}
	else {
		const int BlockSizeSinglePTX = 8;
		const int BlockSizeSinglePTY = 8;
		int blockNumSinglePTX = (cam.resolution.x + BlockSizeSinglePTX - 1) / BlockSizeSinglePTX;
		int blockNumSinglePTY = (cam.resolution.y + BlockSizeSinglePTY - 1) / BlockSizeSinglePTY;

		dim3 singlePTBlockNum(blockNumSinglePTX, blockNumSinglePTY);
		dim3 singlePTBlockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);

		if (Settings::tracer == Tracer::SingleKernel) {
			singleKernelPT<<<singlePTBlockNum, singlePTBlockSize>>>(iter, Settings::traceDepth, hstScene->devScene, cam, devImage);
		}
		else if (Settings::tracer == Tracer::BVHVisualize) {
			BVHVisualize<<<singlePTBlockNum, singlePTBlockSize>>>(iter, hstScene->devScene, cam, devImage);
		}
		else {
			previewGBuffer<<<singlePTBlockNum, singlePTBlockSize>>>(iter, hstScene->devScene, cam, devImage,
				Settings::GBufferPreviewOpt);
		}

		if (guiData != nullptr) {
			guiData->TracedDepth = Settings::traceDepth;
		}
	}

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2D, blockSize2D>>>(pbo, cam.resolution, iter, devImage, Settings::toneMapping);

	// Retrieve image from GPU
	cudaMemcpy(hstScene->state.image.data(), devImage,
		pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathTrace");
}