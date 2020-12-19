#include "CReLUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#pragma warning(pop)

using namespace std;

CReLUFunction::CReLUFunction()
{

}

// Returns a concatenated result of a ReLU that selects only positive predicted with a ReLU that selects only negative predicted
xt::xarray<double> CReLUFunction::CReLU(const xt::xarray<double>& z)
{
	return xt::stack(xt::xtuple(xt::maximum(0.0, z), xt::maximum(0.0, -z)), z.dimension());
}

xt::xarray<double> CReLUFunction::feedForward(const xt::xarray<double>& inputs)
{
	return CReLU(inputs);
}

xt::xarray<double> CReLUFunction::backPropagate(const xt::xarray<double>& sigmas)
{
	auto output = xt::sum(sigmas, -1); // TODO
	return output;
}

xt::xarray<double> CReLUFunction::activationDerivative()
{
	return xt::ones<double>({ 1 });
}

std::vector<size_t> CReLUFunction::getOutputShape()
{
	std::vector<size_t> outputShape;
	outputShape.push_back(0); // TODO
	outputShape.push_back(2);
	return outputShape;
}

void CReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights)
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);

	xt::xarray<double> drawWeights = weights.getParameters();

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = drawWeights(0, i);
		double inv_slope = 1.0 / abs(slope);
		double x1, x2, y1, y2;

		x1 = -min(1.0, inv_slope);
		x2 = min(1.0, inv_slope);
		y1 = abs(x1 * slope);
		y2 = abs(x2 * slope);

		ImVec2 l_start(position.x + (NeuralLayer::DRAW_LEN * x1 * scale), position.y - (NeuralLayer::DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (NeuralLayer::DRAW_LEN * x2 * scale), position.y - (NeuralLayer::DRAW_LEN * y2 * scale));

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