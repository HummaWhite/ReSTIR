#pragma once

#include <iostream>
#include <thrust/random.h>
#include "mathUtil.h"
#include "cudaUtil.h"
#include "common.h"

#if SAMPLER_USE_SOBOL
#define SobolSampleNum 10000
#define SobolSampleDim 200

struct Sampler {
	__device__ Sampler() = default;

	__device__ Sampler(int ptr, uint32_t scramble, uint32_t* data) :
		ptr(ptr), scramble(scramble), data(data) {}

	__device__ float sample() {
		uint32_t r = data[ptr++] ^ scramble;
		scramble = Math::utilhash(scramble);
		return r * 0x1p-32f;
	}

	uint32_t* data;
	uint32_t scramble;
	int ptr;
};

__device__ static Sampler makeSeededRandomEngine(int iter, int index, int dim, uint32_t* data) {
	return Sampler(iter * SobolSampleDim + dim, Math::utilhash(index), data);
}

__device__ inline float sample1D(Sampler& sampler) {
	return sampler.sample();
}

#else
using Sampler = thrust::default_random_engine;

__device__ static Sampler makeSeededRandomEngine(int iter, int index, int dim, uint32_t* data) {
	int h = Math::utilhash((1 << 31) | (dim << 22) | iter) ^ Math::utilhash(index);
	return Sampler(h);
}

__device__ inline float sample1D(Sampler& sampler) {
	return thrust::uniform_real_distribution<float>(0.f, 1.f)(sampler);
}
#endif

__device__ inline glm::vec2 sample2D(Sampler& sampler) {
	return glm::vec2(sample1D(sampler), sample1D(sampler));
}

__device__ inline glm::vec3 sample3D(Sampler& sampler) {
	return glm::vec3(sample2D(sampler), sample1D(sampler));
}

__device__ inline glm::vec4 sample4D(Sampler& sampler) {
	return glm::vec4(sample3D(sampler), sample1D(sampler));
}

template<typename T>
struct BinomialDistrib {
	T prob;
	int failId;
};

/**
* Transform a discrete distribution to a set of binomial distributions
*   so that an O(1) sampling approach can be applied
*/
template<typename T>
struct DiscreteSampler1D {
	using DistribT = BinomialDistrib<T>;

	DiscreteSampler1D() = default;

	DiscreteSampler1D(std::vector<T> values) {
		for (const auto& val : values) {
			sumAll += val;
		}
		T sumInv = static_cast<T>(values.size()) / sumAll;

		for (auto& val : values) {
			val *= sumInv;
		}

		binomDistribs.resize(values.size());
		std::vector<DistribT> stackGtOne(values.size() * 2);
		std::vector<DistribT> stackLsOne(values.size() * 2);
		int topGtOne = 0;
		int topLsOne = 0;

		for (int i = 0; i < values.size(); i++) {
			auto& val = values[i];
			(val > static_cast<T>(1) ? stackGtOne[topGtOne++] : stackLsOne[topLsOne++]) = DistribT{ val, i };
		}

		while (topGtOne && topLsOne) {
			DistribT gt = stackGtOne[--topGtOne];
			DistribT ls = stackLsOne[--topLsOne];

			binomDistribs[ls.failId] = DistribT{ ls.prob, gt.failId };
			// Place ls in the table, and "fill" the rest of probability with gt.prob
			gt.prob -= (static_cast<T>(1) - ls.prob);
			// See if gt.prob is still greater than 1 that it needs more iterations to
			//   be splitted to different binomial distributions
			(gt.prob > static_cast<T>(1) ? stackGtOne[topGtOne++] : stackLsOne[topLsOne++]) = gt;
		}

		for (int i = topGtOne - 1; i >= 0; i--) {
			DistribT gt = stackGtOne[i];
			binomDistribs[gt.failId] = gt;
		}

		for (int i = topLsOne - 1; i >= 0; i--) {
			DistribT ls = stackLsOne[i];
			binomDistribs[ls.failId] = ls;
		}
	}

	void clear() {
		binomDistribs.clear();
		sumAll = static_cast<T>(0);
	}

	int sample(float r1, float r2) {
		int passId = int(float(binomDistribs.size()) * r1);
		DistribT distrib = binomDistribs[passId];
		return (r2 < distrib.prob) ? passId : distrib.failId;
	}

	std::vector<DistribT> binomDistribs;
	T sumAll = static_cast<T>(0);
};

template<typename T>
struct DiscreteSampler2D {
	using DistribT = BinomialDistrib<T>;

	DiscreteSampler2D() = default;

	DiscreteSampler2D(const T* data, int width, int height) {
		columnSamplers.resize(height);
		std::vector<T> sumRows(height);
		std::vector<T> rowData(width);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				sumRows[i] += data[i * width + j];
			}
			float sumRowInv = static_cast<T>(1) / sumRows[i];

			for (int j = 0; j < width; j++) {
				rowData[j] = data[i * width + j] * sumRowInv;
			}
			columnSamplers[i] = DiscreteSampler1D<T>(rowData);
			sumAll += sumRows[i];
		}

		T sumAllInv = static_cast<T>(1) / sumAll;
		for (int i = 0; i < height; i++) {
			sumRows[i] *= sumAllInv;
		}
		rowSampler = DiscreteSampler1D<T>(sumRows);
	}

	void clear() {
		columnSamplers.clear();
		rowSampler.clear();
		sumAll = static_cast<T>(0);
	}

	std::pair<int, int> sample(float r1, float r2, float r3, float r4) {
		int row = rowSampler.sample(r1, r2);
		int column = columnSamplers[row].sample(r3, r4);
		return { row, column };
	}

	std::vector<DiscreteSampler1D<T>> columnSamplers;
	DiscreteSampler1D<T> rowSampler;
	T sumAll = static_cast<T>(0);
};

template<typename T>
struct DevDiscreteSampler1D {
	using DistribT = BinomialDistrib<T>;

	void create(const DiscreteSampler1D<T>& hstSampler) {
		size_t size = byteSizeOf<DistribT>(hstSampler.binomDistribs);
		cudaMalloc(&devBinomDistribs, size);
		cudaMemcpyHostToDev(devBinomDistribs, hstSampler.binomDistribs.data(), size);
		length = hstSampler.binomDistribs.size();
	}

	void destroy() {
		cudaSafeFree(devBinomDistribs);
		length = 0;
	}

	__device__ int sample(float r1, float r2) {
		int passId = glm::min(int(float(length) * r1), length - 1);
		DistribT distrib = devBinomDistribs[passId];
		return (r2 < distrib.prob) ? passId : distrib.failId;
	}

	DistribT* devBinomDistribs = nullptr;
	int length = 0;
};

/**
* Since 2D distribution can be rearranged to 1D distribution,
*   this class is unused
* Sampling 2D distribution with alias sampling table consumes
*   two more random numbers than 1D
*/
template<typename T>
struct DevDiscreteSampler2D {
	using DistribType = BinomialDistrib<T>;

	void create(const std::vector<DiscreteSampler1D<T>>& hstSamplers) {

		width = hstSamplers[0].size();
		height = hstSamplers.size();
	}

	T* devBinomDistribs = nullptr;
	int width = 0;
	int height = 0;
};