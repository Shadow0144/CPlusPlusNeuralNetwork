#pragma once

#include "ParameterizedNeuralLayer.h"
#include "ActivationFunction.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#pragma warning(pop)

class AveragePooling1DNeuralLayer : public ParameterizedNeuralLayer
{
public:
	AveragePooling1DNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape);
	~AveragePooling1DNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	std::vector<size_t> filterShape;

	void draw1DPooling(ImDrawList* canvas, ImVec2 origin, double scale);
};