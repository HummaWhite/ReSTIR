#pragma once
#include <iostream>

#define SAMPLER_USE_SOBOL false

#define SCENE_LIGHT_SINGLE_SIDED true

#define BVH_DISABLE false

#define ENABLE_GBUFFER false

#define CAMERA_PANORAMA false

struct ToneMapping {
    enum {
        None = 0, Filmic = 1, ACES = 2
    };
};

struct Tracer {
    enum {
        Streamed = 0, SingleKernel = 1, BVHVisualize = 2, GBufferPreview = 3
    };
};

struct Settings {
    static int traceDepth;
    static int toneMapping;
    static int tracer;
    static bool sortMaterial;
    static int GBufferPreviewOpt;
};

struct State {
    static bool camChanged;
};
