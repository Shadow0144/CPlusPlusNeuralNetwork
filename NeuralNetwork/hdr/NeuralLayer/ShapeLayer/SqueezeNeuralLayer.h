#pragma once

#include "NeuralLayer/ShapeLayer/ShapeNeuralLayer.h"

#pragma warning(push, 0)
#include <vector>
#pragma warning(pop)

class SqueezeNeuralLayer : public ShapeNeuralLayer
{
public:
	SqueezeNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& squeezeDims = std::vector<size_t>());
	~SqueezeNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	std::vector<size_t> squeezeDims;

	void drawSqueezeFunction(ImDrawList* canvas, ImVec2 origin, double scale);
};