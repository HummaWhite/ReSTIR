#include "restir.h"

#define ReservoirSize 32

using DirectReservoir = Reservoir<DirectLiSample>;
using IndirectReservoir = Reservoir<IndirectLiSample>;

static DirectReservoir* devDirectReservoir = nullptr;
static DirectReservoir* devLastDirectReservoir = nullptr;
static DirectReservoir* devDirectTemp = nullptr;

static IndirectLiSample* devIndLiSample = nullptr;
static IndirectReservoir* devIndTemporalReservoir = nullptr;
static IndirectReservoir* devIndLastTemporalReservoir = nullptr;
static IndirectReservoir* devIndSpatialReservoir = nullptr;
static IndirectReservoir* devIndirectTemp = nullptr;

static bool ReSTIRFirstFrame = true;

template<typename T>
__device__ T findTemporalNeighbor(T* reservoir, int idx, const GBuffer& gBuffer) {
	int primId = gBuffer.primId()[idx];
	int lastIdx = gBuffer.devMotion[idx];
	bool diff = false;

	if (lastIdx < 0) {
		diff = true;
	}
	else if (primId <= NullPrimitive) {
		diff = true;
	}
	else if (gBuffer.lastPrimId()[lastIdx] != primId) {
		diff = true;
	}
	else {
		glm::vec3 norm = DECODE_NORM(gBuffer.normal()[idx]);
		glm::vec3 lastNorm = DECODE_NORM(gBuffer.lastNormal()[lastIdx]);
		float depth = gBuffer.depth()[idx];
		float pdepth = gBuffer.lastDepth()[lastIdx];
		if (Math::absDot(norm, lastNorm) < .9f || glm::abs(pdepth - depth) > depth * .1f) {
			diff = true;
		}
	}
	return diff ? T() : reservoir[lastIdx];
}

template<typename T>
__device__ T findSpatialNeighborDisk(T* reservoir, int x, int y, const GBuffer& gBuffer, glm::vec2 r) {
	const float Radius = 5.f;

	int idx = y * gBuffer.width + x;

	glm::vec2 p = Math::toConcentricDisk(r.x, r.y) * Radius;
	int px = x + .5f + p.x;
	int py = y + .5f + p.y;
	int pidx = py * gBuffer.width + px;

	bool diff = false;

	if (px < 0 || px >= gBuffer.width || py < 0 || py >= gBuffer.height || (px == x && py == y)) {
		diff = true;
	}
	else if (gBuffer.primId()[pidx] != gBuffer.primId()[idx]) {
		diff = true;
	}
	else {
		glm::vec3 norm = DECODE_NORM(gBuffer.normal()[idx]);
		glm::vec3 pnorm = DECODE_NORM(gBuffer.normal()[pidx]);
		if (glm::dot(norm, pnorm) < .9f) {
			diff = true;
		}
#if DENOISER_ENCODE_POSITION
		float depth = gBuffer.depth()[idx];
		float pdepth = gBuffer.depth()[pidx];
		if (glm::abs(depth - pdepth) > depth * .1f) {
#else
		glm::vec3 pos = gBuffer.position()[idx];
		glm::vec3 ppos = gBuffer.position()[pidx];
		if (glm::distance(pos, ppos) > .1f) {
#endif
			diff = true;
		}
	}
	return diff ? T() : reservoir[pidx];
}

__device__ DirectReservoir mergeSpatialNeighborDirect(
	DirectReservoir* reservoirs, int x, int y,
	const GBuffer& gBuffer, Sampler& rng
) {
	DirectReservoir reservoir;
#pragma unroll
	for (int i = 0; i < 5; i++) {
		DirectReservoir spatial = findSpatialNeighborDisk(reservoirs, x, y, gBuffer, sample2D(rng));
		if (!spatial.invalid()) {
			reservoir.merge(spatial, sample1D(rng));
		}
	}
	return reservoir;
}

__device__ glm::vec3 pHatDirect(const DirectReservoir& resv, const Intersection& intersec, const Material& material) {
	DirectLiSample sample = resv.sample;
	return sample.Li * material.BSDF(intersec.norm, intersec.wo, sample.wi) * Math::satDot(intersec.norm, sample.wi);
}

__device__ float bigWDirect(const DirectReservoir& resv, const Intersection& intersec, const Material& material) {
	return resv.weight / (resv.toScalar(pHatDirect(resv, intersec, material)) * static_cast<float>(resv.numSamples));
}

__global__ void ReSTIRDirectKernel(
	int looper, int iter, DevScene* scene, Camera cam,
	glm::vec3* directIllum,
	DirectReservoir* reservoirOut,
	DirectReservoir* reservoirIn,
	DirectReservoir* reservoirTemp,
	GBuffer gBuffer, bool first, int reuseState
) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	glm::vec3 direct(0.f);

	int index = y * cam.resolution.x + x;
	Sampler rng = makeSeededRandomEngine(looper, index, 0, scene->sampleSequence);

	Ray ray = cam.sample(x, y, sample4D(rng));
	Intersection intersec;
	scene->intersect(ray, intersec);

	if (intersec.primId == NullPrimitive) {
		if (scene->envMap != nullptr) {
			direct = scene->envMap->linearSample(Math::toPlane(ray.direction));
		}
		goto WriteRadiance;
	}

	Material material = scene->getTexturedMaterialAndSurface(intersec);
	material.baseColor = glm::vec3(1.f);

	if (material.type == Material::Type::Light) {
		direct = material.baseColor;
		goto WriteRadiance;
	}

	intersec.wo = -ray.direction;

	bool deltaBSDF = (material.type == Material::Type::Dielectric);
	if (!deltaBSDF && glm::dot(intersec.norm, intersec.wo) < 0.f) {
		intersec.norm = -intersec.norm;
	}

	DirectReservoir reservoir;
	for (int i = 0; i < ReservoirSize; i++) {
		glm::vec3 Li;
		glm::vec3 wi;
		float dist;

		float p = scene->sampleDirectLightNoVisibility(intersec.pos, sample4D(rng), Li, wi, dist);
		glm::vec3 g = Li * material.BSDF(intersec.norm, intersec.wo, wi) * Math::satDot(intersec.norm, wi);
		float weight = DirectReservoir::toScalar(g / p);

		if (Math::isNanOrInf(weight) || p <= 0.f) {
			weight = 0.f;
		}
		reservoir.update({ Li, wi, dist }, weight, sample1D(rng));
	}
	DirectLiSample sample = reservoir.sample;

	if (scene->testOcclusion(intersec.pos, intersec.pos + sample.wi * sample.dist)) {
		// Resetting reservoir is INCORRECT
		// Instead, ZERO the weight
		reservoir.weight = 0.f;
	}

	DirectReservoir RISReservoir = reservoir;

	if (!first && (reuseState & ReservoirReuse::Temporal)) {
		DirectReservoir temporal = findTemporalNeighbor(reservoirIn, index, gBuffer);
		if (!temporal.invalid()) {
			reservoir.preClampedMerge<20>(temporal, sample1D(rng));
		}
	}

	sample = reservoir.sample;
	DirectReservoir tempReservoir = reservoir;

	if ((reuseState & ReservoirReuse::Spatial)) {
		reservoir.checkValidity();
		reservoirTemp[index] = reservoir;
		//reservoirTemp[index] = RISReservoir;
		__syncthreads();

		DirectReservoir spatialAggregate = mergeSpatialNeighborDirect(reservoirTemp, x, y, gBuffer, rng);
		if (!spatialAggregate.invalid() && !reservoir.invalid()) {
			reservoir.merge(spatialAggregate, sample1D(rng));
		}

		/*
		__syncthreads();
		reservoirTemp[index] = reservoir;
		__syncthreads();
		spatialAggregate = mergeSpatialNeighborDirect(reservoirTemp, x, y, gBuffer, rng);
		if (!spatialAggregate.invalid()) {
			reservoir.preClampedMerge<4>(spatialAggregate, sample1D(rng));
		}
		*/
	}
	tempReservoir.checkValidity();
	reservoirOut[index] = tempReservoir;
	//reservoir.checkValidity();
	//reservoirOut[index] = reservoir;

	sample = reservoir.sample;
	if (!reservoir.invalid()) {
		/*direct = sample.Li * material.BSDF(intersec.norm, intersec.wo, sample.wi) * Math::satDot(intersec.norm, sample.wi) *
			reservoir.bigW(intersec, material);*/
		glm::vec3 LiBSDF = sample.Li * material.BSDF(intersec.norm, intersec.wo, sample.wi);
		direct = LiBSDF / DirectReservoir::toScalar(LiBSDF) * reservoir.weight / static_cast<float>(reservoir.numSamples);
	}

	if (Math::hasNanOrInf(direct)) {
		direct = glm::vec3(0.f);
	}

WriteRadiance:
	direct *= gBuffer.devAlbedo[index];
	directIllum[index] = (directIllum[index] * float(iter) + direct) / float(iter + 1);
}

__device__ glm::vec3 pHatIndirect(const IndirectLiSample& sample, const Material& material, glm::vec3 wo) {
	return sample.Lo;
	return sample.Lo * material.BSDF(sample.nv, wo, sample.wi()) * Math::satDot(sample.nv, sample.wi());
}

__device__ float bigWIndirect(const IndirectReservoir& resv, const Material& material, glm::vec3 wo) {
	return resv.weight / (resv.toScalar(pHatIndirect(resv.sample, material, wo)) * static_cast<float>(resv.numSamples));
}

__global__ void ReSTIRIndirectKernel(
	int looper, int iter, int maxDepth, DevScene* scene, Camera cam, GBuffer gBuffer,
	IndirectReservoir* temporalReservoir, IndirectReservoir* lastTemporalReservoir,
	glm::vec3* indirectIllum,
	bool first, int reuseState
) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	IndirectLiSample indirectSample;

	int index = y * cam.resolution.x + x;
	Sampler rng = makeSeededRandomEngine(looper, index, 0, scene->sampleSequence);

	Ray ray = cam.sample(x, y, sample4D(rng));
	Intersection intersec;
	scene->intersect(ray, intersec);

	if (intersec.primId == NullPrimitive) {
		goto WriteSample;
	}

	Material material = scene->getTexturedMaterialAndSurface(intersec);

	if (material.type == Material::Type::Light) {
		goto WriteSample;
	}

	glm::vec3 throughput(1.f);
	intersec.wo = -ray.direction;

	float primSamplePdf;
	bool primSampleDelta;
	glm::vec3 primWo = -ray.direction;
	Material primMaterial = material;

	glm::vec3 pathLi(1.f);
	float pathPdf = 1.f;

	for (int depth = 1; depth <= maxDepth; depth++) {
		bool deltaBSDF = (material.type == Material::Type::Dielectric);

		if (material.type != Material::Type::Dielectric && glm::dot(intersec.norm, intersec.wo) < 0.f) {
			intersec.norm = -intersec.norm;
		}

		if (!deltaBSDF && depth > 1) {
			glm::vec3 radiance;
			glm::vec3 wi;
			float lightPdf = scene->sampleDirectLight(intersec.pos, sample4D(rng), radiance, wi);

			if (lightPdf > 0.f) {
				float BSDFPdf = material.pdf(intersec.norm, intersec.wo, wi);
				indirectSample.Lo += throughput * material.BSDF(intersec.norm, intersec.wo, wi) *
					radiance * Math::satDot(intersec.norm, wi) / lightPdf * Math::powerHeuristic(lightPdf, BSDFPdf);
			}
		}

		BSDFSample sample;
		material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);

		if (sample.type == BSDFSampleType::Invalid) {
			break;
		}
		else if (sample.pdf < 1e-8f) {
			break;
		}

		bool deltaSample = (sample.type & BSDFSampleType::Specular);
		if (depth > 1) {
			throughput *= sample.bsdf / sample.pdf * (deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));
			pathLi *= sample.bsdf * (deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));
		}
		else {
			primSamplePdf = sample.pdf;
			primSampleDelta = deltaSample;
			indirectSample.xv = intersec.pos;
			indirectSample.nv = intersec.norm;
		}
		pathPdf *= sample.pdf;

		ray = makeOffsetedRay(intersec.pos, sample.dir);

		glm::vec3 curPos = intersec.pos;
		scene->intersect(ray, intersec);
		intersec.wo = -ray.direction;

		if (intersec.primId == NullPrimitive) {
			//break;
			if (scene->envMap != nullptr) {
				glm::vec3 radiance = scene->envMap->linearSample(Math::toPlane(ray.direction))
					* throughput;

				float weight = deltaSample ? 1.f :
					Math::powerHeuristic(sample.pdf, scene->environmentMapPdf(ray.direction));
				indirectSample.Lo += radiance * weight;
			}
			break;
		}
		material = scene->getTexturedMaterialAndSurface(intersec);

		if (material.type == Material::Type::Light) {
			if (glm::dot(intersec.norm, ray.direction) < 0.f) {
#if SCENE_LIGHT_SINGLE_SIDED
				break;
#else
				intersec.norm = -intersec.norm;
#endif
			}
			glm::vec3 radiance = material.baseColor;

			float weight = (deltaSample || depth == 1) ? 1.f : Math::powerHeuristic(
				sample.pdf,
				Math::pdfAreaToSolidAngle(Math::luminance(radiance) * scene->sumLightPowerInv *
					scene->getPrimitiveArea(intersec.primId), curPos, intersec.pos, intersec.norm)
			);
			indirectSample.Lo += radiance * throughput * weight;

			if (depth == 1) {
				indirectSample.xs = intersec.pos;
				indirectSample.ns = intersec.norm;
			}
			break;
		}

		if (depth == 1) {
			indirectSample.xs = intersec.pos;
			indirectSample.ns = intersec.norm;
		}
	}
WriteSample:
	IndirectReservoir reservoir;
	float sampleWeight = 0.f;
	if (!indirectSample.invalid()) {
		sampleWeight = IndirectReservoir::toScalar(pHatIndirect(indirectSample, primMaterial, primWo) / primSamplePdf);
		if (isnan(sampleWeight) || sampleWeight < 0.f) {
			// !!!!!!  CRITICAL  !!!!!!
			// If the sample candidate is invalid, you should SET WEIGHT TO ZERO instead of clearing the reservoir
			sampleWeight = 0.f;
		}
	}
	reservoir.update(indirectSample, sampleWeight, sample1D(rng));

	if (!first && (reuseState & ReservoirReuse::Temporal)) {
		IndirectReservoir tempReservoir = findTemporalNeighbor(lastTemporalReservoir, index, gBuffer);
		if (!tempReservoir.invalid()) {
			reservoir.merge(tempReservoir, sample1D(rng));
		}
	}
	glm::vec3 indirect(0.f);

	IndirectLiSample sample = reservoir.sample;
	
	reservoir.clamp<20>();

	if (!reservoir.invalid()) {
		glm::vec3 primWi = glm::normalize(sample.xs - sample.xv);
		
		/*indirect = reservoir.sample.Lo * primMaterial.BSDF(sample.nv, primWo, primWi) * Math::satDot(sample.nv, primWi) *
			bigWIndirect(reservoir, primMaterial, primWo);*/
		
		indirect = reservoir.sample.Lo / IndirectReservoir::toScalar(reservoir.sample.Lo) * reservoir.weight / static_cast<float>(reservoir.numSamples);
		indirect *= primMaterial.BSDF(sample.nv, primWo, primWi) * (primSampleDelta ? 1.f : Math::satDot(sample.nv, primWi));
		
	}
	
	/*
	glm::vec3 primWi = glm::normalize(sample.xs - sample.xv);
	indirect = sample.Lo * primMaterial.BSDF(sample.nv, primWo, primWi) *
		(primSampleDelta ? 1.f : Math::absDot(sample.nv, primWi)) / primSamplePdf;
		*/
		

	if (Math::hasNanOrInf(indirect)) {
		indirect = glm::vec3(0.f);
	}

	temporalReservoir[index] = reservoir;
	indirectIllum[index] = (indirectIllum[index] * float(iter) + indirect) / float(iter + 1);
}

void ReSTIRDirect(glm::vec3* devDirectIllum, int iter, const GBuffer& gBuffer) {
	const Camera& cam = State::scene->camera;

	const int BlockSizeSinglePTX = 8;
	const int BlockSizeSinglePTY = 8;
	int blockNumSinglePTX = ceilDiv(cam.resolution.x, BlockSizeSinglePTX);
	int blockNumSinglePTY = ceilDiv(cam.resolution.y, BlockSizeSinglePTY);

	dim3 blockNum(blockNumSinglePTX, blockNumSinglePTY);
	dim3 blockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);

	ReSTIRDirectKernel<<<blockNum, blockSize>>>(
		State::looper, iter, State::scene->devScene, cam, devDirectIllum,
		devDirectReservoir, devLastDirectReservoir, devDirectTemp, gBuffer, ReSTIRFirstFrame,
		Settings::reservoirReuse
	);
	std::swap(devDirectReservoir, devLastDirectReservoir);

	if (ReSTIRFirstFrame) {
		ReSTIRFirstFrame = false;
	}

	checkCUDAError("ReSTIR Direct");
#if SAMPLER_USE_SOBOL
	State::looper = (State::looper + 1) % SobolSampleNum;
#else
	State::looper++;
#endif
}

void ReSTIRIndirect(glm::vec3* devIndirectIllum, int iter, const GBuffer& gBuffer) {
	const Camera& cam = State::scene->camera;

	const int BlockSizeSinglePTX = 8;
	const int BlockSizeSinglePTY = 8;
	int blockNumSinglePTX = ceilDiv(cam.resolution.x, BlockSizeSinglePTX);
	int blockNumSinglePTY = ceilDiv(cam.resolution.y, BlockSizeSinglePTY);

	dim3 blockNum(blockNumSinglePTX, blockNumSinglePTY);
	dim3 blockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);

	ReSTIRIndirectKernel<<<blockNum, blockSize>>>(
		State::looper, iter, Settings::traceDepth, State::scene->devScene, cam, gBuffer,
		devIndTemporalReservoir, devIndLastTemporalReservoir,
		devIndirectIllum, ReSTIRFirstFrame, Settings::reservoirReuse
	);
	std::swap(devIndTemporalReservoir, devIndLastTemporalReservoir);

	if (ReSTIRFirstFrame) {
		ReSTIRFirstFrame = false;
	}

	checkCUDAError("ReSTIR Indirect");
#if SAMPLER_USE_SOBOL
	State::looper = (State::looper + 1) % SobolSampleNum;
#else
	State::looper++;
#endif
}

void ReSTIRInit() {
	const Camera& cam = State::scene->camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	devDirectReservoir = cudaMalloc<DirectReservoir>(pixelcount);
	cudaMemset(devDirectReservoir, 0, pixelcount * sizeof(DirectReservoir));

	devLastDirectReservoir = cudaMalloc<DirectReservoir>(pixelcount);
	cudaMemset(devLastDirectReservoir, 0, pixelcount * sizeof(DirectReservoir));

	devDirectTemp = cudaMalloc<DirectReservoir>(pixelcount);
	cudaMemset(devDirectTemp, 0, pixelcount * sizeof(DirectReservoir));

	devIndLiSample = cudaMalloc<IndirectLiSample>(pixelcount);

	devIndTemporalReservoir = cudaMalloc<IndirectReservoir>(pixelcount);
	cudaMemset(devIndTemporalReservoir, 0, pixelcount * sizeof(IndirectReservoir));
	devIndLastTemporalReservoir = cudaMalloc<IndirectReservoir>(pixelcount);
	cudaMemset(devIndLastTemporalReservoir, 0, pixelcount * sizeof(IndirectReservoir));

	/*
	static IndirectReservoir* devIndTemporalReservoir = nullptr;
	static IndirectReservoir* devLastIndTemporalReservoir = nullptr;
	static IndirectReservoir* devIndSpatialReservoir = nullptr;
	static IndirectReservoir* devIndirectTemp = nullptr;
	*/
}

void ReSTIRFree() {
	cudaSafeFree(devDirectReservoir);
	cudaSafeFree(devLastDirectReservoir);
	cudaSafeFree(devDirectTemp);
	cudaSafeFree(devIndLiSample);

	cudaSafeFree(devIndTemporalReservoir);
	cudaSafeFree(devIndLastTemporalReservoir);
}

void ReSTIRReset() {
	ReSTIRFirstFrame = true;
}