#pragma once

#include <vector>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

static void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

template<typename T>
T* cudaMalloc(size_t numElements) {
    T* devPtr;
    cudaMalloc(&devPtr, sizeof(T) * numElements);
    return devPtr;
}

template<typename T>
void cudaSafeFree(T*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

template<typename T>
size_t byteSizeOf(const std::vector<T>& v) {
    return v.size() * sizeof(T);
}

static cudaError_t __stdcall cudaMemcpyHostToDev(void* dev, const void* host, size_t size) {
    return cudaMemcpy(dev, host, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
}

static cudaError_t __stdcall cudaMemcpyDevToDev(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

static cudaError_t __stdcall cudaMemcpyDevToHost(void* host, const void* dev, size_t size) {
    return cudaMemcpy(host, dev, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}