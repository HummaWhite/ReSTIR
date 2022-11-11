#include "denoiser.h"

/*__device__ constexpr float Gaussian3x3[] = {
	.25f, .5f, .25f
};

__device__ constexpr float Gaussian5x5[] = {
	.0625f, .25f, .375f, .25f, .0625f
};*/

__device__ constexpr float Gaussian3x3[3][3] = {
	{ .075f, .124f, .075f },
	{ .124f, .204f, .124f },
	{ .075f, .124f, .075f }
};

__device__ constexpr float Gaussian5x5[5][5] = {
	{ .0030f, .0133f, .0219f, .0133f, .0030f },
	{ .0133f, .0596f, .0983f, .0596f, .0133f },
	{ .0219f, .0983f, .1621f, .0983f, .0219f },
	{ .0133f, .0596f, .0983f, .0596f, .0133f },
	{ .0030f, .0133f, .0219f, .0133f, .0030f }
};

#if DENOISER_ENCODE_NORMAL
#  define ENCODE_NORM(x) Math::encodeNormalHemiOct32(x)
#  define DECODE_NORM(x) Math::decodeNormalHemiOct32(x)
#else
#  define ENCODE_NORM(x) x
#  define DECODE_NORM(x) x
#endif

__global__ void renderGBuffer(DevScene* scene, Camera cam, GBuffer gBuffer) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	int idx = y * cam.resolution.x + x;

	float aspect = float(cam.resolution.x) / cam.resolution.y;
	float tanFovY = glm::tan(glm::radians(cam.fov.y));
	glm::vec2 pixelSize = 1.f / glm::vec2(cam.resolution);
	glm::vec2 scr = glm::vec2(x, y) * pixelSize;
	glm::vec2 ruv = scr + pixelSize * glm::vec2(.5f);

	glm::vec3 pLens(0.f);
	glm::vec3 pFocusPlane = glm::vec3((1.f - ruv * 2.f) * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
	glm::vec3 dir = pFocusPlane - pLens;

	Ray ray;
	ray.direction = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);
	ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;

	Intersection intersec;
	scene->intersect(ray, intersec);

	if (intersec.primId != NullPrimitive) {
		int matId = intersec.matId;
		if (scene->materials[intersec.matId].type == Material::Type::Light) {
			matId = NullPrimitive - 1;
#if SCENE_LIGHT_SINGLE_SIDED
			if (glm::dot(intersec.norm, ray.direction) < 0.f) {
				intersec.primId = NullPrimitive;
			}
#endif
		}
		Material material = scene->getTexturedMaterialAndSurface(intersec);

		gBuffer.devAlbedo[idx] = material.baseColor;
		gBuffer.normal()[idx] = ENCODE_NORM(intersec.norm);
		gBuffer.primId()[idx] = matId;
#if DENOISER_ENCODE_POSITION
		gBuffer.depth()[idx] = glm::distance(intersec.pos, ray.origin);
#else
		gBuffer.position()[idx] = intersec.pos;
#endif

		glm::ivec2 lastPos = gBuffer.lastCamera.getRasterCoord(intersec.pos);
		if (lastPos.x >= 0 && lastPos.x < gBuffer.width && lastPos.y >= 0 && lastPos.y < gBuffer.height) {
			gBuffer.devMotion[idx] = lastPos.y * cam.resolution.x + lastPos.x;
		}
		else {
			gBuffer.devMotion[idx] = -1;
		}
	}
	else {
		glm::vec3 albedo(0.f);
		if (scene->envMap != nullptr) {
			glm::vec2 uv = Math::toPlane(ray.direction);
			albedo = scene->envMap->linearSample(uv);
		}
		gBuffer.devAlbedo[idx] = albedo;
		gBuffer.normal()[idx] = GBuffer::NormT(0.f);
		gBuffer.primId()[idx] = NullPrimitive;
#if DENOISER_ENCODE_POSITION
		gBuffer.depth()[idx] = 1.f;
#else
		gBuffer.position()[idx] = glm::vec3(0.f);
#endif
		gBuffer.devMotion[idx] = 0;
	}
}

__global__ void naiveWaveletFilter(
	glm::vec3* devColorOut, glm::vec3* devColorIn, GBuffer gBuffer,
	float sigDepth, float sigNormal, float sigLuminance, Camera cam, int level
) {
	int step = 1 << level;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	int idxP = y * cam.resolution.x + x;
	int primIdP = gBuffer.primId()[idxP];

	glm::vec3 colorP = devColorIn[idxP];

	glm::vec3 sum(0.f);
	float sumWeight = 0.f;
#pragma unroll
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			int qx = x + j * step;
			int qy = y + i * step;
			int idxQ = qy * cam.resolution.x + qx;

			if (qx >= cam.resolution.x || qy >= cam.resolution.y ||
				qx < 0 || qy < 0) {
				continue;
			}

			float weight = Gaussian5x5[i + 2][j + 2];
			sum += devColorIn[idxQ] * weight;
			sumWeight += weight;
		}
	}
	devColorOut[idxP] = (sumWeight == 0.f) ? devColorIn[idxP] : sum / sumWeight;
}

__global__ void waveletFilter(
	glm::vec3* devColorOut, glm::vec3* devColorIn, GBuffer gBuffer,
	float sigDepth, float sigNormal, float sigLuminance, Camera cam, int level
) {
	int step = 1 << level;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	int idxP = y * cam.resolution.x + x;
	int primIdP = gBuffer.primId()[idxP];

	if (primIdP <= NullPrimitive) {
		devColorOut[idxP] = devColorIn[idxP];
		return;
	}

	glm::vec3 normP = DECODE_NORM(gBuffer.normal()[idxP]);
	glm::vec3 colorP = devColorIn[idxP];
	glm::vec3 posP =
#if DENOISER_ENCODE_POSITION
		cam.getPosition(x, y, gBuffer.depth()[idxP]);
#else
		gBuffer.position()[idxP];
#endif

	glm::vec3 sum(0.f);
	float sumWeight = 0.f;
#pragma unroll
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			int qx = x + j * step;
			int qy = y + i * step;
			int idxQ = qy * cam.resolution.x + qx;

			if (qx >= cam.resolution.x || qy >= cam.resolution.y ||
				qx < 0 || qy < 0) {
				continue;
			}

			if (gBuffer.primId()[idxQ] != primIdP) {
				continue;
			}
			glm::vec3 normQ = DECODE_NORM(gBuffer.normal()[idxQ]);
			glm::vec3 colorQ = devColorIn[idxQ];
			glm::vec3 posQ =
#if DENOISER_ENCODE_POSITION
				cam.getPosition(qx, qy, gBuffer.depth()[idxQ]);
#else
				gBuffer.position()[idxQ];
#endif

			float distColor2 = glm::dot(colorP - colorQ, colorP - colorQ);
			float wColor = glm::min(1.f, glm::exp(-distColor2 / sigLuminance));

			float distNorm2 = glm::dot(normP - normQ, normP - normQ);
			float wNorm = glm::min(1.f, glm::exp(-distNorm2 / sigNormal));

			float distPos2 = glm::dot(posP - posQ, posP - posQ);
			float wPos = glm::min(1.f, glm::exp(-distPos2 / sigDepth));

			float weight = wColor * wNorm * wPos * Gaussian5x5[i + 2][j + 2];
			sum += colorQ * weight;
			sumWeight += weight;
		}
	}
	devColorOut[idxP] = (sumWeight == 0.f) ? devColorIn[idxP] : sum / sumWeight;
}

/*
* SVGF version, filtering variance at the same time
*/
__global__ void waveletFilter(
	glm::vec3* devColorOut, glm::vec3* devColorIn, float* devVarianceOut, float* devVarainceIn, float* devVarFiltered,
	GBuffer gBuffer, float sigDepth, float sigNormal, float sigLuminance, Camera cam, int level
) {
	int step = 1 << level;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= cam.resolution.x || y >= cam.resolution.y) {
		return;
	}
	int idxP = y * cam.resolution.x + x;
	int primIdP = gBuffer.primId()[idxP];

	if (primIdP <= NullPrimitive) {
		devColorOut[idxP] = devColorIn[idxP];
		devVarianceOut[idxP] = devVarainceIn[idxP];
		return;
	}

	glm::vec3 normP = DECODE_NORM(gBuffer.normal()[idxP]);
	glm::vec3 colorP = devColorIn[idxP];
	glm::vec3 posP =
#if DENOISER_ENCODE_POSITION
		cam.getPosition(x, y, gBuffer.depth()[idxP]);
#else
		gBuffer.position()[idxP];
#endif

	glm::vec3 sumColor(0.f);
	float sumVariance = 0.f;
	float sumWeight = 0.f;
	float sumWeight2 = 0.f;
#pragma unroll
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			int qx = x + j * step;
			int qy = y + i * step;
			int idxQ = qy * cam.resolution.x + qx;

			if (qx >= cam.resolution.x || qy >= cam.resolution.y ||
				qx < 0 || qy < 0) {
				continue;
			}
			if (gBuffer.primId()[idxQ] != primIdP) {
				continue;
			}
			glm::vec3 normQ = DECODE_NORM(gBuffer.normal()[idxQ]);
			glm::vec3 colorQ = devColorIn[idxQ];
			glm::vec3 posQ =
#if DENOISER_ENCODE_POSITION
				cam.getPosition(qx, qy, gBuffer.depth()[idxQ]);
#else
				gBuffer.position()[idxQ];
#endif
			float varQ = devVarainceIn[idxQ];

			float distPos2 = glm::dot(posP - posQ, posP - posQ);
			float wPos = glm::exp(-distPos2 / sigDepth) + 1e-4f;

			float wNorm = glm::pow(Math::satDot(normP, normQ), sigNormal) + 1e-4f;

			float denom = sigLuminance * glm::sqrt(glm::max(devVarFiltered[idxQ], 0.f)) + 1e-4f;
			float wColor = glm::exp(-glm::abs(Math::luminance(colorP) - Math::luminance(colorQ)) / denom) + 1e-4f;

			float weight = wColor * wNorm * wPos * Gaussian5x5[i + 2][j + 2];
			float weight2 = weight * weight;

			sumColor += colorQ * weight;
			sumVariance += varQ * weight2;
			sumWeight += weight;
			sumWeight2 += weight2;
		}
	}
	devColorOut[idxP] = (sumWeight < FLT_EPSILON) ? devColorIn[idxP] : sumColor / sumWeight;
	devVarianceOut[idxP] = (sumWeight2 < FLT_EPSILON) ? devVarainceIn[idxP] : sumVariance / sumWeight2;
}

__global__ void modulate(glm::vec3* devImage, GBuffer gBuffer, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = y * width + x;
		glm::vec3 color = devImage[idx];
		color = Math::LDRToHDR(color);
		devImage[idx] = color * glm::max(gBuffer.devAlbedo[idx]/* - DEMODULATE_EPS*/, glm::vec3(0.f));
	}
}

__global__ void add(glm::vec3* devImage, glm::vec3* devIn, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = y * width + x;
		devImage[idx] += devIn[idx];
	}
}

__global__ void add(glm::vec3* devOut, glm::vec3* devIn1, glm::vec3* devIn2, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < width && y < height) {
		int idx = y * width + x;
		devOut[idx] = devIn1[idx] + devIn2[idx];
	}
}

__global__ void temporalAccumulate(
	glm::vec3* devColorAccumOut, glm::vec3* devColorAccumIn, 
	glm::vec3* devMomentAccumOut, glm::vec3* devMomentAccumIn,
	glm::vec3* devColorIn,
	GBuffer gBuffer, bool first
) {
	const float Alpha = .2f;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= gBuffer.width || y >= gBuffer.height) {
		return;
	}
	int idx = y * gBuffer.width + x;
	int primId = gBuffer.primId()[idx];
	int lastIdx = gBuffer.devMotion[idx];

	bool diff = first;

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
		if (glm::abs(glm::dot(norm, lastNorm)) < .1f) {
			diff = true;
		}
	}

	glm::vec3 color = devColorIn[idx];
	glm::vec3 lastColor = devColorAccumIn[lastIdx];
	glm::vec3 lastMoment = devMomentAccumIn[lastIdx];
	float lum = Math::luminance(color);

	glm::vec3 accumColor;
	glm::vec3 accumMoment;

	if (diff) {
		accumColor = color;
		accumMoment = { lum, lum * lum, 0.f };
	}
	else {
		accumColor = glm::mix(lastColor, color, Alpha);
		accumMoment = glm::vec3(glm::mix(glm::vec2(lastMoment), glm::vec2(lum, lum * lum), Alpha), lastMoment.b + 1.f);
	}
	devColorAccumOut[idx] = accumColor;
	devMomentAccumOut[idx] = accumMoment;
}

__global__ void estimateVariance(float* devVariance, glm::vec3* devMoment, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
	int idx = y * width + x;

	glm::vec3 m = devMoment[idx];
	if (m.z > 3.5f) {
		// Temporal variance
		devVariance[idx] = m.y - m.x * m.x;
	}
	else {
		// Spatial variance
		glm::vec2 sumMoment(0.f);
		int numPixel = 0;

		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				int qx = x + j;
				int qy = y + i;

				if (qx < 0 || qx >= width || qy < 0 || qy >= height) {
					continue;
				}
				int idxQ = qy * width + qx;

				sumMoment += glm::vec2(devMoment[idxQ]);
				numPixel++;
			}
		}
		sumMoment /= numPixel;
		devVariance[idx] = sumMoment.y - sumMoment.x * sumMoment.x;
	}
}

__global__ void filterVariance(float* devVarianceOut, float* devVarianceIn, int width, int height) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
	int idx = y * width + x;

	float sum = 0.f;
	float sumWeight = 0.f;
#pragma unroll
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			int qx = x + i;
			int qy = y + j;
			if (qx < 0 || qx >= width || qy < 0 || qy >= height) {
				continue;
			}
			int idxQ = qy * width + qx;
			float weight = Gaussian3x3[i + 1][j + 1];
			sum += devVarianceIn[idxQ] * weight;
			sumWeight += weight;
		}
	}
	devVarianceOut[idx] = sum / sumWeight;
}

void GBuffer::create(int width, int height) {
	this->width = width;
	this->height = height;
	int numPixels = width * height;
	devAlbedo = cudaMalloc<glm::vec3>(numPixels);
	devMotion = cudaMalloc<int>(numPixels);

	for (int i = 0; i < 2; i++) {
		devNormal[i] = cudaMalloc<NormT>(numPixels);
		devPrimId[i] = cudaMalloc<int>(numPixels);
#if DENOISER_ENCODE_POSITION
		devDepth[i] = cudaMalloc<float>(numPixels);
#else
		devPosition[i] = cudaMalloc<glm::vec3>(numPixels);
#endif
	}
}

void GBuffer::destroy() {
	cudaSafeFree(devAlbedo);
	cudaSafeFree(devMotion);
	for (int i = 0; i < 2; i++) {
		cudaSafeFree(devNormal[i]);
		cudaSafeFree(devPrimId[i]);
#if DENOISER_ENCODE_POSITION
		cudaSafeFree(devDepth[i]);
#else
		cudaSafeFree(devPosition[i]);
#endif
	}
}

void GBuffer::update(const Camera& cam) {
	lastCamera = cam;
	frameIdx ^= 1;
}

void GBuffer::render(DevScene* scene, const Camera& cam) {
	constexpr int BlockSize = 8;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(cam.resolution.x, BlockSize), ceilDiv(cam.resolution.y, BlockSize));
	renderGBuffer<<<blockNum, blockSize>>>(scene, cam, *this);
	checkCUDAError("renderGBuffer");
}

void modulateAlbedo(glm::vec3* devImage, const GBuffer& gBuffer) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(gBuffer.width, BlockSize), ceilDiv(gBuffer.height, BlockSize));
	modulate<<<blockNum, blockSize>>>(devImage, gBuffer, gBuffer.width, gBuffer.height);
	checkCUDAError("modulate");
}

void addImage(glm::vec3* devImage, glm::vec3* devIn, int width, int height) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	add<<<blockNum, blockSize>>>(devImage, devIn, width, height);
}

void addImage(glm::vec3* devOut, glm::vec3* devIn1, glm::vec3* devIn2, int width, int height) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	add<<<blockNum, blockSize>>>(devOut, devIn1, devIn2, width, height);
}

void EAWaveletFilter::filter(
	glm::vec3* devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam, int level
) {
	constexpr int BlockSize = 8;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	waveletFilter<<<blockNum, blockSize>>>(
		devColorOut, devColorIn, gBuffer, sigDepth, sigNormal, sigLumin, cam, level
	);
	checkCUDAError("EAW Filter");
}

void EAWaveletFilter::filter(
	glm::vec3* devColorOut, glm::vec3* devColorIn,
	float* devVarianceOut, float* devVarianceIn, float* devFilteredVar,
	const GBuffer& gBuffer, const Camera& cam, int level
) {
	constexpr int BlockSize = 16;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(width, BlockSize), ceilDiv(height, BlockSize));
	waveletFilter<<<blockNum, blockSize>>>(
		devColorOut, devColorIn, devVarianceOut, devVarianceIn, devFilteredVar,
		gBuffer, sigDepth, sigNormal, sigLumin, cam, level
	);
}

void LeveledEAWFilter::create(int width, int height, int level) {
	this->level = level;
	waveletFilter = EAWaveletFilter(width, height, 64.f, .2f, 1.f);
	devTempImg = cudaMalloc<glm::vec3>(width * height);
}

void LeveledEAWFilter::destroy() {
	cudaSafeFree(devTempImg);
}

void LeveledEAWFilter::filter(glm::vec3*& devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam) {
	waveletFilter.filter(devColorOut, devColorIn, gBuffer, cam, 0);

	waveletFilter.filter(devTempImg, devColorOut, gBuffer, cam, 1);
	std::swap(devColorOut, devTempImg);

	waveletFilter.filter(devTempImg, devColorOut, gBuffer, cam, 2);
	std::swap(devColorOut, devTempImg);

	waveletFilter.filter(devTempImg, devColorOut, gBuffer, cam, 3);
	std::swap(devColorOut, devTempImg);

	waveletFilter.filter(devTempImg, devColorOut, gBuffer, cam, 4);
	std::swap(devColorOut, devTempImg);
}

void SpatioTemporalFilter::create(int width, int height, int level) {
	this->level = level;

	for (int i = 0; i < 2; i++) {
		devAccumColor[i] = cudaMalloc<glm::vec3>(width * height);
		devAccumMoment[i] = cudaMalloc<glm::vec3>(width * height);
	}

	devVariance = cudaMalloc<float>(width * height);
	waveletFilter = EAWaveletFilter(width, height, 4.f, 128.f, 1.f);

	devTempColor = cudaMalloc<glm::vec3>(width * height);
	devTempVariance = cudaMalloc<float>(width * height);
	devFilteredVariance = cudaMalloc<float>(width * height);
}

void SpatioTemporalFilter::destroy() {
	for (int i = 0; i < 2; i++) {
		cudaSafeFree(devAccumColor[i]);
		cudaSafeFree(devAccumMoment[i]);
	}
	cudaSafeFree(devVariance);
	cudaSafeFree(devTempColor);
	cudaSafeFree(devTempVariance);
	cudaSafeFree(devFilteredVariance);
}

void SpatioTemporalFilter::temporalAccumulate(glm::vec3* devColorIn, const GBuffer& gBuffer) {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(gBuffer.width, BlockSize), ceilDiv(gBuffer.height, BlockSize));

	::temporalAccumulate<<<blockNum, blockSize>>>(
		devAccumColor[frameIdx], devAccumColor[frameIdx ^ 1],
		devAccumMoment[frameIdx], devAccumMoment[frameIdx ^ 1],
		devColorIn, gBuffer, firstTime
	);

	firstTime = false;
	checkCUDAError("SpatioTemporalFilter::temporalAccumulate");
}

void SpatioTemporalFilter::estimateVariance() {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(waveletFilter.width, BlockSize), ceilDiv(waveletFilter.height, BlockSize));
	::estimateVariance<<<blockNum, blockSize>>>(devVariance, devAccumMoment[frameIdx], waveletFilter.width, waveletFilter.height);
	checkCUDAError("SpatioTemporalFilter::estimateVariance");
}

void SpatioTemporalFilter::filterVariance() {
	constexpr int BlockSize = 32;
	dim3 blockSize(BlockSize, BlockSize);
	dim3 blockNum(ceilDiv(waveletFilter.width, BlockSize), ceilDiv(waveletFilter.height, BlockSize));
	::filterVariance<<<blockNum, blockSize>>>(devFilteredVariance, devVariance, waveletFilter.width, waveletFilter.height);
	checkCUDAError("SpatioTemporalFilter::filterVariance");
}

void SpatioTemporalFilter::filter(glm::vec3*& devColorOut, glm::vec3* devColorIn, const GBuffer& gBuffer, const Camera& cam) {
	temporalAccumulate(devColorIn, gBuffer);
	estimateVariance();

	filterVariance();
	waveletFilter.filter(devColorOut, devAccumColor[frameIdx], devTempVariance, devVariance, devFilteredVariance, gBuffer, cam, 0);
	std::swap(devColorOut, devAccumColor[frameIdx]);
	std::swap(devTempVariance, devVariance);

	filterVariance();
	waveletFilter.filter(devColorOut, devAccumColor[frameIdx], devTempVariance, devVariance, devFilteredVariance, gBuffer, cam, 1);
	std::swap(devTempVariance, devVariance);

	filterVariance();
	waveletFilter.filter(devTempColor, devColorOut, devTempVariance, devVariance, devFilteredVariance, gBuffer, cam, 2);
	std::swap(devTempColor, devColorOut);
	std::swap(devTempVariance, devVariance);

	filterVariance();
	waveletFilter.filter(devTempColor, devColorOut, devTempVariance, devVariance, devFilteredVariance, gBuffer, cam, 3);
	std::swap(devTempColor, devColorOut);
	std::swap(devTempVariance, devVariance);

	filterVariance();
	waveletFilter.filter(devTempColor, devColorOut, devTempVariance, devVariance, devFilteredVariance, gBuffer, cam, 4);
	std::swap(devTempColor, devColorOut);
	std::swap(devTempVariance, devVariance);
}

void SpatioTemporalFilter::nextFrame() {
	frameIdx ^= 1;
}