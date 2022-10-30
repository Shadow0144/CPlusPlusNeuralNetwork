#include "ReLU6Function.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

ReLU6Function::ReLU6Function()
{

}

xt::xarray<double> ReLU6Function::reLU6(const xt::xarray<double>& z) const
{
	return xt::minimum(xt::maximum(0.0, z), 6.0);
}

xt::xarray<double> ReLU6Function::feedForward(const xt::xarray<double>& inputs) const
{
	return reLU6(inputs);
}

xt::xarray<double> ReLU6Function::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return (sigmas * ((lastOutput > 0.0) < 6.0));
}

void ReLU6Function::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
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
		double inv_slope = (slope == 0) ? (0.0) : (1.0 / abs(slope));
		double x1, x2, y1, y2;
		if (slope > 0.0)
		{
			x1 = -1.0;
			x2 = +min(1.0, inv_slope);
			y1 = 0.0;
			y2 = min((x2 * slope), 6.0);
		}
		else
		{
			x1 = -min(1.0, inv_slope);
			x2 = 1.0;
			y1 = min((x1 * slope), 6.0);
			y2 = 0.0;
		}

		ImVec2 l_start(position.x + (NeuralLayer::DRAW_LEN * x1 * scale), position.y - (NeuralLayer::DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (NeuralLayer::DRAW_LEN * x2 * scale), position.y - (NeuralLayer::DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}