#include "PReLUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <random>
#include <xtensor/xrandom.hpp>
#pragma warning(pop)

using namespace std;

PReLUFunction::PReLUFunction()
{
	this->a = 2.0 * (xt::random::rand<double>({ 1 }) - 0.5)(0);
	this->deltaA = 0.0;
}

xt::xarray<double> PReLUFunction::PReLU(const xt::xarray<double>& z)
{
	return xt::maximum(a * z, z);
}

xt::xarray<double> PReLUFunction::feedForward(const xt::xarray<double>& inputs)
{
	return PReLU(inputs);
}

xt::xarray<double> PReLUFunction::backPropagate(const xt::xarray<double>& sigmas)
{
	/*auto mask = (lastOutput <= 0.0);
	deltaA += xt::sum<double>(lastOutput * mask)();
	return denseBackpropagate(sigmas * activationDerivative());*/
	return sigmas;
}

double PReLUFunction::applyBackPropagate() // Returns the sum of the change in the weights
{
	//a += -ALPHA * deltaA;
	return 0;// ActivationFunction::applyBackPropagate();
}

xt::xarray<double> PReLUFunction::getGradient(const xt::xarray<double>& sigmas)
{
	auto mask = (lastOutput > 0.0);
	return (mask + (a * (xt::ones<double>(mask.shape()) - mask)));
}

void PReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
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
		double x1, x2, y1, y2;
		if (slope > 0.0f)
		{
			x1 = -1.0;
			x2 = +min(1.0, inv_slope);
			y1 = -a;
			y2 = (x2 * slope);
		}
		else
		{
			x1 = -min(1.0, inv_slope);
			x2 = 1.0;
			y1 = (x1 * slope);
			y2 = a;
		}

		ImVec2 l_start(position.x + (NeuralLayer::DRAW_LEN * x1 * scale), position.y - (NeuralLayer::DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (NeuralLayer::DRAW_LEN * x2 * scale), position.y - (NeuralLayer::DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}