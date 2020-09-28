#include "AbsoluteReLUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

AbsoluteReLUFunction::AbsoluteReLUFunction(size_t incomingUnits, size_t numUnits)
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

xt::xarray<double> AbsoluteReLUFunction::absoluteReLU(xt::xarray<double> z)
{
	return xt::abs(z);
}

xt::xarray<double> AbsoluteReLUFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	lastOutput = absoluteReLU(dotProductResult);
	return lastOutput;
}

xt::xarray<double> AbsoluteReLUFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> AbsoluteReLUFunction::activationDerivative()
{
	auto mask = (lastOutput > 0.0);
	return (mask + (mask - (xt::ones<double>(mask.shape()))));
}

void AbsoluteReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = abs(weights.getParameters()(0, i));
		double inv_slope = 1.0 / abs(slope);
		double x1, x2, y1, y2;
		x1 = -min(1.0, inv_slope);
		x2 = +min(1.0, inv_slope);
		y1 = abs(x1 * slope);
		y2 = abs(x2 * slope);

		ImVec2 l_start(position.x + (DRAW_LEN * x1 * scale), position.y - (DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (DRAW_LEN * x2 * scale), position.y - (DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}