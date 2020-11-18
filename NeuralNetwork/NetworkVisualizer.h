#pragma once

#pragma warning(push, 0)
#include <SDL.h>
#include <thread>
#include "imgui.h"
#include <shared_mutex>
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

	bool getThreadRunning();

	void setTargets(xt::xarray<double> inputs, xt::xarray<double> targets);
	void setPredicted(xt::xarray<double> predicted);

private:
	void setup();
	void draw();
	void renderFrame();

	NeuralNetwork* network;
	bool windowClosed;
	bool rendering;
	bool threadRunning;
	mutable std::shared_mutex resultsMutex;

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

	thread drawThread;

	xt::xarray<double> inputs;
	xt::xarray<double> predicted;
	xt::xarray<double> targets;
	bool inputsSet;
	bool predictedSet;
	bool targetsSet;
};