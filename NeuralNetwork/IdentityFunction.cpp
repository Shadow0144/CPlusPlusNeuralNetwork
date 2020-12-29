#include "IdentityFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xview.hpp>
#pragma warning(pop)

using namespace std;

IdentityFunction::IdentityFunction()
{

}

xt::xarray<double> IdentityFunction::feedForward(const xt::xarray<double>& inputs)
{
	return inputs;
}

xt::xarray<double> IdentityFunction::getGradient(const xt::xarray<double>& sigmas)
{
	return sigmas;
}

void IdentityFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

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

		ImVec2 l_start(position.x + (x1 * NeuralLayer::DRAW_LEN * scale), position.y - (y1 * NeuralLayer::DRAW_LEN * scale));
		ImVec2 l_end(position.x + (x2 * NeuralLayer::DRAW_LEN * scale), position.y - (y2 * NeuralLayer::DRAW_LEN * scale));

		canvas->AddLine(l_start, l_end, BLACK);
	}
}