#pragma once

#include "PoolingNeuralLayer.h"
#include "ActivationFunction.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#pragma warning(pop)

class AveragePooling1DNeuralLayer : public PoolingNeuralLayer
{
public:
	AveragePooling1DNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape, bool hasChannels = true);
	~AveragePooling1DNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

private:
	void drawPooling(ImDrawList* canvas, ImVec2 origin, double scale);
};