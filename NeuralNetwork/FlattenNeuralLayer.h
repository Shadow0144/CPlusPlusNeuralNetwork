#pragma once

#include "ShapeNeuralLayer.h"

#pragma warning(push, 0)
#include <vector>
#pragma warning(pop)

class FlattenNeuralLayer : public ShapeNeuralLayer
{
public:
	FlattenNeuralLayer(NeuralLayer* parent, int numOutputs);
	~FlattenNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	void drawFlattenFunction(ImDrawList* canvas, ImVec2 origin, double scale);
};