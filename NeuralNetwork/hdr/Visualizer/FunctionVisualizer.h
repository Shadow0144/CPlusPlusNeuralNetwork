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

	void draw(ImDrawList* canvas, const xt::xarray<double>& inputs, const xt::xarray<double>& targets);

private:
	NetworkVisualizer* visualizer;
	NeuralNetwork* network;
};