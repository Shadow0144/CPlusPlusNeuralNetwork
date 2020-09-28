#include "PReLUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <random>
#include <xtensor/xrandom.hpp>
#pragma warning(pop)

using namespace std;

PReLUFunction::PReLUFunction(size_t incomingUnits, size_t numUnits)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// input x output -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
	this->a = 2.0 * (xt::random::rand<double>({ 1 }) - 0.5)(0);
	this->deltaA = 0.0;
}

xt::xarray<double> PReLUFunction::PReLU(xt::xarray<double> z)
{
	return xt::maximum(a * z, z);
}

xt::xarray<double> PReLUFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	lastOutput = PReLU(dotProductResult);
	return lastOutput;
}

xt::xarray<double> PReLUFunction::backPropagate(xt::xarray<double> sigmas)
{
	auto mask = (lastOutput <= 0.0);
	deltaA += xt::sum<double>(lastOutput * mask)();
	return denseBackpropagate(sigmas * activationDerivative());
}

double PReLUFunction::applyBackPropagate() // Returns the sum of the change in the weights
{
	a += -ALPHA * deltaA;
	return Function::applyBackPropagate();
}

xt::xarray<double> PReLUFunction::activationDerivative()
{
	auto mask = (lastOutput > 0.0);
	return (mask + (a * (xt::ones<double>(mask.shape()) - mask)));
}

void PReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = weights.getParameters()(0, i);
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

		ImVec2 l_start(position.x + (DRAW_LEN * x1 * scale), position.y - (DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (DRAW_LEN * x2 * scale), position.y - (DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}