#include "gbuffer.h"

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