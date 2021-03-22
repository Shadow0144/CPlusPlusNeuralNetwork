#pragma once

#include "ParameterizedNeuralLayer.h"
#include "ActivationFunction.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#pragma warning(pop)

class MaxPooling2DNeuralLayer : public ParameterizedNeuralLayer
{
public:
	MaxPooling2DNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape);
	~MaxPooling2DNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	std::vector<size_t> filterShape;

	void draw2DPooling(ImDrawList* canvas, ImVec2 origin, double scale);
};