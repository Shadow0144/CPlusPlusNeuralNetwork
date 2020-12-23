#include "PReLUFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <random>
#include <xtensor/xrandom.hpp>
#pragma warning(pop)

#include "Test.h"

using namespace std;

PReLUFunction::PReLUFunction(int numUnits)
{
	this->a = xt::random::rand<double>({ numUnits }); // Strictly positive values
	this->deltaA = xt::zeros<double>({ numUnits });
}

xt::xarray<double> PReLUFunction::PReLU(const xt::xarray<double>& z)
{
	auto nMask = (z <= 0.0);
	auto pMask = (z > 0.0);
	return (a * z * nMask) + (z * pMask);
}

xt::xarray<double> PReLUFunction::feedForward(const xt::xarray<double>& inputs)
{
	return PReLU(inputs);
}

void PReLUFunction::applyBackPropagate()
{
	a -= 0.01 * deltaA; // TODO: Fix ALPHA
	
}

xt::xarray<double> PReLUFunction::getGradient(const xt::xarray<double>& sigmas)
{
	auto nMask = (lastInput <= 0.0);
	auto pMask = (lastInput > 0.0);
	std::vector<size_t> dims;
	int DIMS = lastInput.dimension() - 1;
	for (int i = 0; i < DIMS; i++)
	{
		dims.push_back(i);
	}
	deltaA = xt::sum<double>(lastInput * nMask, dims) / lastInput.shape()[0];
	return (sigmas * (pMask + (a * (xt::ones<double>(pMask.shape()) - pMask))));
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
		double inv_slope = (slope == 0) ? (0.0) : (1.0 / abs(slope));
		double x1, x2, y1, y2;
		if (slope > 0.0f)
		{
			x1 = max(-1.0, -a(i) * inv_slope);
			x2 = min(+1.0, +inv_slope);
			y1 = (-a(i) * slope);
			y2 = (x2 * slope);
		}
		else
		{
			x1 = max(-1.0, -inv_slope);
			x2 = min(+1.0, +a(i) * inv_slope);
			y1 = (x1 * slope);
			y2 = (a(i) * slope);
		}

		ImVec2 l_start(position.x + (NeuralLayer::DRAW_LEN * x1 * scale), position.y - (NeuralLayer::DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (NeuralLayer::DRAW_LEN * x2 * scale), position.y - (NeuralLayer::DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}