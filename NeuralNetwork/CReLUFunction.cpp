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
// Note: This increases the dimension by one
xt::xarray<double> CReLUFunction::CReLU(const xt::xarray<double>& z) const
{
	return xt::stack(xt::xtuple(xt::maximum(0.0, z), xt::maximum(0.0, -z)), z.dimension());
}

xt::xarray<double> CReLUFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return CReLU(inputs);
}

xt::xarray<double> CReLUFunction::getGradient(const xt::xarray<double>& sigmas) const
{
	xt::xarray<double> newSigmas = sigmas;
	xt::strided_view(newSigmas, { xt::ellipsis(), 1 }) *= -1;
	return xt::sum(newSigmas, -1);
}

std::vector<size_t> CReLUFunction::getOutputShape(std::vector<size_t> outputShape) const
{
	outputShape.push_back(2);
	return outputShape;
}

void CReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor DARK_RED(0.3f, 0.0f, 0.0f, 1.0f);

	xt::xarray<double> drawWeights = weights.getParameters();

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = drawWeights(0, i);
		double inv_slope = (slope == 0) ? (0.0) : (1.0 / abs(slope));
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
			canvas->AddLine(l_mid, l_end, DARK_RED);
		}
		else
		{
			canvas->AddLine(l_start, l_mid, DARK_RED);
			canvas->AddLine(l_mid, l_end, BLACK);
		}
	}
}