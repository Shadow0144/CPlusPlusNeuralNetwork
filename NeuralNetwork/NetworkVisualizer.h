#pragma once

#pragma warning(push, 0)
#include <SDL.h>
#include "imgui.h"
#pragma warning(pop)

#include "FunctionVisualizer.h"
#include "ClassifierVisualizer.h"

class NeuralNetwork;

class NetworkVisualizer
{
public:
	NetworkVisualizer(NeuralNetwork* network);
	~NetworkVisualizer();

	bool getWindowClosed();

	void addFunctionVisualization();
	void addClassificationVisualization(int rows, int cols, ImColor* classColors);

	ImVec2 getWindowSize();

	void draw(xt::xarray<double> inputs = NULL, xt::xarray<double> targets = NULL);

private:
	void setup();

	NeuralNetwork* network;
	bool windowClosed;

	SDL_Window* window;
	SDL_GLContext gl_context;
	ImGuiIO io;

	ClassifierVisualizer* classifier;
	bool displayClasses;

	FunctionVisualizer* function;
	bool displayFunctions;

	ImVec2 winSize;
	ImVec2 origin;
	ImVec2 drag;
	bool startDrag;
	double scale;
	const double SCALE_FACTOR = 0.1;
	const double MIN_SCALE_FACTOR = 0.1;
	const double MAX_SCALE_FACTOR = 10.0;
};