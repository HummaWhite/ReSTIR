#pragma once

#include "common.h"

extern GLuint pbo;
extern Scene* hstScene;

std::string currentTimeString();
bool init();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);