#include "common.h"

int Settings::traceDepth = 0;
int Settings::toneMapping = ToneMapping::ACES;
int Settings::tracer = Tracer::Streamed;
int Settings::ImagePreviewOpt = 8;
int Settings::denoiser = Denoiser::None;
bool Settings::modulate = true;
bool Settings::animateCamera = false;
float Settings::animateRadius = 1.f;
float Settings::animateSpeed = 2.7f;
float Settings::meshLightSampleWeight = 1.f;
bool Settings::useReservoir = true;
int Settings::reservoirReuse = ReservoirReuse::Temporal;
bool Settings::accumulate = false;

bool State::camChanged = true;
int State::looper = 0;
Scene* State::scene = nullptr;