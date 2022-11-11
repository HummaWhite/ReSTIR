//#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h"
#include "preview.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

std::string currentTimeString() {
	time_t now;
	time(&now);
	char buf[sizeof "0000-00-00_00-00-00z"];
	strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
	return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures() {
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
	GLfloat vertices[] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f,  1.0f,
		-1.0f,  1.0f,
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
	const char* attribLocations[] = { "Position", "Texcoords" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	//glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	return program;
}

void deletePBO(GLuint* pbo) {
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void deleteTexture(GLuint* tex) {
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void cleanupCuda() {
	if (pbo) {
		deletePBO(&pbo);
	}
	if (displayImage) {
		deleteTexture(&displayImage);
	}
}

void initCuda() {
	cudaGLSetGLDevice(0);
	checkCUDAError("cudaGLSetDevice");

	// Clean up on program exit
	atexit(cleanupCuda);
}

void initPBO() {
	// set up vertex data parameter
	int num_texels = width * height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);

}

void errorCallback(int error, const char* description) {
	fprintf(stderr, "%s\n", description);
}

bool init() {
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetScrollCallback(window, mouseScrollCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}
	//printf("Opengl Version:%s\n", glGetString(GL_VERSION));
	//Set up ImGui

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	io = &ImGui::GetIO(); (void)io;
	ImGui::StyleColorsClassic();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 450");

	auto& guiStyle = ImGui::GetStyle();
	guiStyle.FrameRounding = 1.0f;
	guiStyle.FramePadding.y = 2.0f;
	guiStyle.ItemSpacing.y = 6.0f;
	guiStyle.GrabRounding = 1.0f;

	// Initialize other stuff
	initVAO();
	initTextures();
	initCuda();
	initPBO();
	GLuint passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	return true;
}

void InitImguiData(GuiDataContainer* guiData) {
	imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui() {
	mouseOverImGuiWinow = io->WantCaptureMouse;

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	static float f = 0.0f;
	static int counter = 0;

	ImGui::Begin("Path Tracer Analytics", nullptr,
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoDecoration |
		ImGuiWindowFlags_AlwaysAutoResize |
		ImGuiWindowFlags_NoSavedSettings |
		ImGuiWindowFlags_NoFocusOnAppearing |
		ImGuiWindowFlags_NoNav); {
		ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
		ImGui::Text("BVH Size %d", scene->BVHSize);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		ImGui::End();
	}

	ImGui::Begin("Options"); {
		const char* Denoisers[] = { "None", "EA A-Trous", "SVGF" };
		if (ImGui::Combo("Denoiser", &Settings::denoiser, Denoisers, IM_ARRAYSIZE(Denoisers))) {
			State::camChanged = true;
		}

		ImGui::Checkbox("Animate Camera", &Settings::animateCamera);
		if (Settings::animateCamera) {
			State::camChanged = true;
			ImGui::SliderFloat("Radius", &Settings::animateRadius, 0.f, 10.f);
			ImGui::SliderFloat("Speed", &Settings::animateSpeed, 0.1f, 10.f);
		}
		
		ImGui::Checkbox("Modulate", &Settings::modulate);

		if (Settings::denoiser == Denoiser::EAWavelet) {
			const char* Modes[] = {
				"Albedo", "Normal", "Depth/Position", "Motion",
				"Input Direct", "Input Indirect",
				"Output Direct", "Output Indirect",
				"Composed"
			};
			ImGui::Combo("Preview", &Settings::ImagePreviewOpt, Modes, IM_ARRAYSIZE(Modes));

			ImGui::SliderInt("Levels", &EAWFilter.level, 1, 5);
			ImGui::DragFloat("Sigma Lumin", &EAWFilter.waveletFilter.sigLumin, .01f, 0.f);
			ImGui::DragFloat("Sigma Normal", &EAWFilter.waveletFilter.sigNormal, .01f, 0.f);
			ImGui::DragFloat("Sigma Depth", &EAWFilter.waveletFilter.sigDepth, .01f, 0.f);
		}
		else if (Settings::denoiser == Denoiser::SVGF) {
			const char* Modes[] = {
				"Albedo", "Normal", "Depth/Position", "Motion",
				"Input Direct", "Input Indirect",
				"Output Direct", "Output Indirect",
				"Composed",
				"Direct Moment", "Indirect Moment",
				"Direct Variance", "Indirect Variance"
			};
			ImGui::Combo("Preview", &Settings::ImagePreviewOpt, Modes, IM_ARRAYSIZE(Modes));

			ImGui::SliderInt("Levels", &directFilter.level, 1, 5);
			ImGui::DragFloat("Sigma Lumin", &directFilter.waveletFilter.sigLumin, .01f, 0.f);
			ImGui::DragFloat("Sigma Normal", &directFilter.waveletFilter.sigNormal, .01f, 0.f);
			ImGui::DragFloat("Sigma Depth", &directFilter.waveletFilter.sigDepth, .01f, 0.f);

			indirectFilter.level = directFilter.level;
			indirectFilter.waveletFilter.sigLumin = directFilter.waveletFilter.sigLumin;
			indirectFilter.waveletFilter.sigNormal = directFilter.waveletFilter.sigNormal;
			indirectFilter.waveletFilter.sigDepth = directFilter.waveletFilter.sigDepth;
		}

		if (ImGui::InputInt("Max Depth", &Settings::traceDepth, 1, 1)) {
			State::camChanged = true;
		}
		ImGui::Separator();

		ImGui::Text("Post Processing");
		const char* ToneMappingMethods[] = { "None", "Filmic", "ACES" };
		ImGui::Combo("Tone Mapping", &Settings::toneMapping, ToneMappingMethods, IM_ARRAYSIZE(ToneMappingMethods));
		ImGui::Separator();

		ImGui::End();
	}

	ImGui::Begin("Camera"); {
		auto& cam = scene->camera;

		glm::vec3 lastPos = cam.position;
		if (ImGui::DragFloat3("Position", glm::value_ptr(cam.position), .1f)) {
			State::camChanged = true;
		}

		if (ImGui::DragFloat3("Rotation", glm::value_ptr(cam.rotation), .1f)) {
			State::camChanged = true;
		}

		if (ImGui::SliderFloat("FOV", &cam.fov.y, .1f, 45.f, "%.1f deg")) {
			State::camChanged = true;
		}

		if (ImGui::DragFloat("Aperture", &cam.lensRadius, .01f, 0.f, 3.f)) {
			State::camChanged = true;
		}
		if (ImGui::DragFloat("Focal", &cam.focalDist, .1f, 0.f, FLT_MAX, "%.1f m")) {
			State::camChanged = true;
		}

		ImGui::End();
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow() {
	return mouseOverImGuiWinow;
}

void mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		
		glfwPollEvents();

		runCuda();

		std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
		glfwSetWindowTitle(window, title.c_str());
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// Binding GL_PIXEL_UNPACK_BUFFER back to default
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		// Render ImGui Stuff
		RenderImGui();

		glfwSwapBuffers(window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}
