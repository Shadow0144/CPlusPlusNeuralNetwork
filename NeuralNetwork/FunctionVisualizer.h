#pragma once

#pragma warning(push, 0)
#include "NeuralNetwork.h"
#include <xtensor/xarray.hpp>
#include "imgui.h"
#pragma warning(pop)

class NetworkVisualizer;

class FunctionVisualizer
{
public:
	FunctionVisualizer(NetworkVisualizer* visualizer, NeuralNetwork* network);
	~FunctionVisualizer();

	void draw(ImDrawList* canvas, xt::xarray<double> inputs, xt::xarray<double> targets);

private:
	NetworkVisualizer* visualizer;
	NeuralNetwork* network;
};