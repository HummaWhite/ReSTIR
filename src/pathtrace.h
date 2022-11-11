#pragma once

#include <device_launch_parameters.h>
#include <vector>
#include "scene.h"
#include "common.h"

void InitDataContainer(GuiDataContainer* guiData);

void copyImageToPBO(uchar4* devPBO, glm::vec3* devImage, int width, int height, int toneMapping, float scale = 1.f);
void copyImageToPBO(uchar4* devPBO, glm::vec2* devImage, int width, int height);
void copyImageToPBO(uchar4* devPBO, float* devImage, int width, int height);
void copyImageToPBO(uchar4* devPBO, int* devImage, int width, int height);

void pathTraceInit(Scene *scene);
void pathTraceFree();
void pathTrace(glm::vec3* devDirectIllum, glm::vec3* devIndirectIllum, int iter);