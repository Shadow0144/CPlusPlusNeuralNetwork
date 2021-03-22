#pragma once

#include "ParameterizedNeuralLayer.h"
#include "ActivationFunction.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#pragma warning(pop)

class AveragePooling2DNeuralLayer : public ParameterizedNeuralLayer
{
public:
	AveragePooling2DNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape);
	~AveragePooling2DNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	std::vector<size_t> filterShape;

	void draw2DPooling(ImDrawList* canvas, ImVec2 origin, double scale);
};