#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "glslUtility.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>

#include "scene.h"
#include "denoiser.h"

extern Scene* scene;
extern int iteration;
extern int width;
extern int height;
extern LeveledEAWFilter EAWFilter;
extern SpatioTemporalFilter directFilter;
extern SpatioTemporalFilter indirectFilter;

void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mouseScrollCallback(GLFWwindow* window, double offsetX, double offsetY);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);