#include "AbsoluteReLUFunction.h"
#include "NeuralNetwork.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

AbsoluteReLUFunction::AbsoluteReLUFunction()
{
}

xt::xarray<double> AbsoluteReLUFunction::absoluteReLU(const xt::xarray<double>& z)
{
	return xt::abs(z);
}

xt::xarray<double> AbsoluteReLUFunction::feedForward(const xt::xarray<double>& inputs)
{
	return absoluteReLU(inputs);
}

xt::xarray<double> AbsoluteReLUFunction::getGradient(const xt::xarray<double>& sigmas)
{
	auto mask = (lastOutput > 0.0);
	return (sigmas * (mask + (mask - (xt::ones<double>(mask.shape())))));
}

void AbsoluteReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	xt::xarray<double> drawWeights = weights.getParameters();

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = abs(drawWeights(0, i));
		double inv_slope = 1.0 / abs(slope);
		double x1, x2, y1, y2;
		x1 = -min(1.0, inv_slope);
		x2 = +min(1.0, inv_slope);
		y1 = abs(x1 * slope);
		y2 = abs(x2 * slope);

		ImVec2 l_start(position.x + (NeuralLayer::DRAW_LEN * x1 * scale), position.y - (NeuralLayer::DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (NeuralLayer::DRAW_LEN * x2 * scale), position.y - (NeuralLayer::DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}