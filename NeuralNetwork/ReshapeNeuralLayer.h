#pragma once

#include "ShapeNeuralLayer.h"

#pragma warning(push, 0)
#include <vector>
#pragma warning(pop)

class ReshapeNeuralLayer : public ShapeNeuralLayer
{
public:
	ReshapeNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& newShape);
	~ReshapeNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	std::vector<size_t> newShape;

	void drawReshapeFunction(ImDrawList* canvas, ImVec2 origin, double scale);
};