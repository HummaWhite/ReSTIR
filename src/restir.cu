#include "restir.h"

#define ReservoirSize 32

using DirectReservoir = Reservoir<LightLiSample>;

static DirectReservoir* devDirectReservoir = nullptr;
static DirectReservoir* devLastDirectReservoir = nullptr;
static DirectReservoir* devDirectTemp = nullptr;
static bool ReSTIRFirstFrame = true;

__device__ DirectReservoir mergeReservoir(const DirectReservoir& a, const DirectReservoir& b, glm::vec2 r) {
	DirectReservoir reservoir;
	reservoir.update(a.sample, a.weight, r.x);
	reservoir.update(b.sample, b.weight, r.y);
	reservoir.numSamples = a.numSamples + b.numSamples;
	return reservoir;
}

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
		if (Math::absDot(norm, lastNorm) < .1f) {
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
		if (glm::dot(norm, pnorm) < .1f) {
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
	LightLiSample sample = reservoir.sample;

	if (scene->testOcclusion(intersec.pos, intersec.pos + sample.wi * sample.dist)) {
		// Resetting reservoir is INCORRECT
		// Instead, ZERO the weight
		reservoir.weight = 0.f;
	}

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

__global__ void spatialReuseDirect(
	int looper, int iter,
	DirectReservoir* reservoirOut, DirectReservoir* reservoirIn,
	DevScene* scene, GBuffer gBuffer
) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= gBuffer.width || y >= gBuffer.height) {
		return;
	}
	int index = y * gBuffer.width + x;

	Sampler rng = makeSeededRandomEngine(looper, index, 5 * ReservoirSize + 1, scene->sampleSequence);
	DirectReservoir reservoir = reservoirIn[index];

	if (reservoir.numSamples == 0) {
		reservoirOut[index].clear();
		return;
	}

#pragma unroll
	for (int i = 0; i < 5; i++) {
		DirectReservoir neighbor = findSpatialNeighborDisk(reservoirIn, x, y, gBuffer, sample2D(rng));
		if (!neighbor.invalid()) {
			reservoir.merge(neighbor, sample1D(rng));
		}
	}
	reservoirOut[index] = reservoir;
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

void ReSTIRInit() {
	const Camera& cam = State::scene->camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	devDirectReservoir = cudaMalloc<DirectReservoir>(pixelcount);
	cudaMemset(devDirectReservoir, 0, pixelcount * sizeof(DirectReservoir));
	devLastDirectReservoir = cudaMalloc<DirectReservoir>(pixelcount);
	cudaMemset(devLastDirectReservoir, 0, pixelcount * sizeof(DirectReservoir));
	devDirectTemp = cudaMalloc<DirectReservoir>(pixelcount);
	cudaMemset(devDirectTemp, 0, pixelcount * sizeof(DirectReservoir));
}

void ReSTIRFree() {
	cudaSafeFree(devDirectReservoir);
	cudaSafeFree(devLastDirectReservoir);
	cudaSafeFree(devDirectTemp);
}