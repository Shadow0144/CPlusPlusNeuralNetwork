#include "TanhFunction.h"
#include "NeuralLayer.h"

#include <iostream>

using namespace std;

TanhFunction::TanhFunction(size_t incomingUnits, size_t numUnits)
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

xt::xarray<double> TanhFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	lastOutput = xt::tanh(dotProductResult);
	return lastOutput;
}

xt::xarray<double> TanhFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> TanhFunction::activationDerivative()
{
	return (1.0 - xt::pow(lastOutput, 2));
}

void TanhFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	const int R = 3;
	const double RANGE = 3.0;

	const int RESOLUTION = (R * 4) + 1;
	MatrixXd sP(RESOLUTION, 2);
	for (int r = 0; r < RESOLUTION; r++)
	{
		sP(r, 0) = RANGE * (2.0 * r) / (RESOLUTION - 1.0) - RANGE;
	}

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		for (int r = 0; r < RESOLUTION; r++)
		{
			sP(r, 1) = tanh(sP(r, 0) * weights.getParameters()(0, i));
		}

		float rescale = (1.0 / RANGE) * DRAW_LEN * scale;
		for (int d = 0; d < (RESOLUTION - 3); d += 3)
		{
			MatrixXd points = approximateBezier(sP.block(d, 0, 4, 2));
			canvas->AddBezierCurve(
				ImVec2(position.x + (points(0, 0) * rescale), position.y - (points(0, 1) * rescale)),
				ImVec2(position.x + (points(1, 0) * rescale), position.y - (points(1, 1) * rescale)),
				ImVec2(position.x + (points(2, 0) * rescale), position.y - (points(2, 1) * rescale)),
				ImVec2(position.x + (points(3, 0) * rescale), position.y - (points(3, 1) * rescale)),
				BLACK, 1);
		}
	}
}