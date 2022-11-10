#pragma once

#include <device_launch_parameters.h>
#include <vector>
#include "scene.h"
#include "common.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathTraceInit(Scene *scene);
void pathTraceFree();
void pathTrace(uchar4 *pbo, int frame, int iteration);