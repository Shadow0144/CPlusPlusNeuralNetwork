#pragma once

#pragma warning(push, 0)
#include <SDL.h>
#include "imgui.h"
#pragma warning(pop)

class NeuralNetwork;

class NetworkVisualizer
{
public:
	NetworkVisualizer(NeuralNetwork* network);
	~NetworkVisualizer();

	bool getWindowClosed();

	void draw();
private:
	void setup();

	NeuralNetwork* network;
	bool windowClosed;

	SDL_Window* window;
	SDL_GLContext gl_context;
	ImGuiIO io;

	ImVec2 origin;
	ImVec2 drag;
	bool startDrag;
	float scale;
	const float SCALE_FACTOR = 0.1f;
};