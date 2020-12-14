#pragma once

#include "NeuralLayer.h"
#include "Function.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#pragma warning(pop)

class MaxPooling2DNeuralLayer : public NeuralLayer
{
public:
	MaxPooling2DNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape);
	~MaxPooling2DNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	std::vector<size_t> filterShape;

	ParameterSet weights;

	void draw2DPooling(ImDrawList* canvas, ImVec2 origin, double scale);
};