#pragma once

#include "NeuralLayer/PoolingLayer/PoolingNeuralLayer.h"
#include "ActivationFunction/ActivationFunction.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#pragma warning(pop)

class MaxPooling2DNeuralLayer : public PoolingNeuralLayer
{
public:
	MaxPooling2DNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape, 
							const std::vector<size_t>& stride = { }, bool hasChannels = true);
	~MaxPooling2DNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

private:
	void drawPooling(ImDrawList* canvas, ImVec2 origin, double scale);
};