#include "common.h"

int Settings::traceDepth = 0;
int Settings::toneMapping = ToneMapping::ACES;
int Settings::tracer = Tracer::Streamed;
int Settings::ImagePreviewOpt = 8;
int Settings::denoiser = Denoiser::SVGF;
bool Settings::modulate = true;
bool Settings::animateCamera = false;
float Settings::animateRadius = 1.f;
float Settings::animateSpeed = 2.7f;

bool State::camChanged = true;