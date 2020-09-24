#include "CReLUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

// TODO: Concatenate!
CReLUFunction::CReLUFunction(size_t incomingUnits, size_t numUnits)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// incoming x current -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
}

// Returns a concatenated result of a ReLU that selects only positive results with a ReLU that selects only negative results
xt::xarray<double> CReLUFunction::CReLU(xt::xarray<double> z)
{
	return xt::stack(xt::xtuple(xt::maximum(0.0, z), xt::maximum(0.0, -z)), z.dimension());
}

xt::xarray<double> CReLUFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	lastOutput = CReLU(dotProductResult);
	return lastOutput;
}

xt::xarray<double> CReLUFunction::backPropagate(xt::xarray<double> sigmas)
{
	sigmas = xt::sum(sigmas, -1);
	return denseBackpropagate(sigmas);
}

xt::xarray<double> CReLUFunction::activationDerivative()
{
	return xt::ones<double>({ 1 });
}

std::vector<size_t> CReLUFunction::getOutputShape()
{
	std::vector<size_t> outputShape;
	outputShape.push_back(numUnits);
	outputShape.push_back(2);
	return outputShape;
}

void CReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = weights.getParameters()(0, i);
		double inv_slope = 1.0 / abs(slope);
		double x1, x2, y1, y2;

		x1 = -min(1.0, inv_slope);
		x2 = min(1.0, inv_slope);
		y1 = abs(x1 * slope);
		y2 = abs(x2 * slope);

		ImVec2 l_start(position.x + (DRAW_LEN * x1 * scale), position.y - (DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (DRAW_LEN * x2 * scale), position.y - (DRAW_LEN * y2 * scale));

		if (slope < 0.0)
		{
			canvas->AddLine(l_start, l_mid, BLACK);
			canvas->AddLine(l_mid, l_end, LIGHT_GRAY);
		}
		else
		{
			canvas->AddLine(l_start, l_mid, LIGHT_GRAY);
			canvas->AddLine(l_mid, l_end, BLACK);
		}
	}
}