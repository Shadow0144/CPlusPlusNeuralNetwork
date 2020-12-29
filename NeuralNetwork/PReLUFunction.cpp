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
	const double ALPHA = 0.01;
	a -= ALPHA * deltaA; // TODO: Fix ALPHA
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
	const ImColor DARK_DULL_RED(0.6f, 0.3f, 0.3f, 1.0f);

	xt::xarray<double> drawWeights = weights.getParameters();

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = drawWeights(0, i);
		double aSlope = a(i) * slope;
		double x, y, ax, ay;

		// Line of primary ReLU
		if (slope > 0.0 && slope < 1.0)
		{
			x = +1.0;
			y = slope;
		}
		else if (slope > 0.0 && slope >= 1.0)
		{
			x = (slope != 0.0) ? (1.0 / slope) : 0.0;
			y = +1.0;
		}
		else if (slope <= 0.0 && slope > -1.0)
		{
			x = -1.0;
			y = slope;
		}
		else
		{
			x = (slope != 0.0) ? (1.0 / slope) : 0.0;
			y = -1.0;
		}

		// Line of weighted negative ReLU
		if (aSlope > 0.0 && aSlope < 1.0)
		{
			ax = -1.0;
			ay = -aSlope;
		}
		else if (aSlope > 0.0 && aSlope >= 1.0)
		{
			ax = (aSlope != 0.0) ? (-1.0 / aSlope) : 0.0;
			ay = -1.0;
		}
		else if (aSlope <= 0.0 && aSlope > -1.0)
		{
			ax = +1.0;
			ay = -aSlope;
		}
		else
		{
			ax = (aSlope != 0.0) ? (-1.0 / aSlope) : 0.0;
			ay = +1.0;
		}

		ImVec2 l_start(position.x + (NeuralLayer::DRAW_LEN * ax * scale), position.y - (NeuralLayer::DRAW_LEN * ay * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (NeuralLayer::DRAW_LEN * x * scale), position.y - (NeuralLayer::DRAW_LEN * y * scale));

		canvas->AddLine(l_start, l_mid, DARK_DULL_RED);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}