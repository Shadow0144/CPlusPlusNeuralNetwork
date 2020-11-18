#include "LinearFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xview.hpp>
#pragma warning(pop)

using namespace std;

LinearFunction::LinearFunction(size_t incomingUnits, size_t numUnits)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// input x output -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
}

xt::xarray<double> LinearFunction::feedForward(xt::xarray<double> inputs)
{
	return dotProduct(inputs);
}

xt::xarray<double> LinearFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas);
}

void LinearFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	xt::xarray<double> drawWeights = weights.getParameters();

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		double slope = drawWeights(0, i);
		double inv_slope = 1.0 / abs(slope);
		double x1 = -min(1.0, inv_slope);
		double x2 = +min(1.0, inv_slope);
		double y1 = x1 * slope;
		double y2 = x2 * slope;

		ImVec2 l_start(position.x + (x1 * DRAW_LEN * scale), position.y - (y1 * DRAW_LEN * scale));
		ImVec2 l_end(position.x + (x2 * DRAW_LEN * scale), position.y - (y2 * DRAW_LEN * scale));

		canvas->AddLine(l_start, l_end, BLACK);
	}
}