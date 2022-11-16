#pragma once
#include <iostream>

#define SAMPLER_USE_SOBOL false

#define SCENE_LIGHT_SINGLE_SIDED true

#define DENOISER_DEMODULATE true
#define DENOISER_ENCODE_NORMAL false
#define DENOISER_ENCODE_POSITION true

#define DEMODULATE_EPS 1e-3f

#define DenoiseClamp 128.f
#define DenoiseCompress 16.f
#define DenoiseLightId -2

struct ToneMapping {
    enum {
        None = 0, Filmic = 1, ACES = 2
    };
};

struct Tracer {
    enum {
        Streamed = 0, SingleKernel = 1, BVHVisualize = 2, GBufferPreview = 3, ReSTIRDI = 4
    };
};

struct Denoiser {
    enum {
        None, EAWavelet, SVGF
    };
};

struct ReservoirReuse {
    enum {
        None = 0b00,
        Temporal = 0b01,
        Spatial = 0b10,
        Spatiotemporal = 0b11,
    };
};

struct Scene;

struct Settings {
    static int traceDepth;
    static int toneMapping;
    static int tracer;
    static int ImagePreviewOpt;
    static int denoiser;
    static bool modulate;
    static bool animateCamera;
    static float animateRadius;
    static float animateSpeed;
    static float meshLightSampleWeight;
    static bool useReservoir;
    static int reservoirReuse;
    static bool accumulate;
};

struct State {
    static bool camChanged;
    static int looper;
    static Scene* scene;
};
