#include "SoftplusFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

SoftplusFunction::SoftplusFunction(size_t incomingUnits, size_t numUnits)
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

double SoftplusFunction::softplus(double z)
{
	return (log(1.0 + exp(k * z)) / k);
}

xt::xarray<double> SoftplusFunction::softplus(xt::xarray<double> z)
{
	return (log(1.0 + exp(k * z)) / k);
}

xt::xarray<double> SoftplusFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	return softplus(dotProductResult);
}

xt::xarray<double> SoftplusFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> SoftplusFunction::activationDerivative()
{
	return (1.0 / (1.0 + exp(-k * xt::linalg::tensordot(lastInput, weights.getParameters(), 1))));
}

double SoftplusFunction::getK() 
{
	return k;
}

void SoftplusFunction::setK(double k)
{
	this->k = k;
}

void SoftplusFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
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
			sP(r, 1) = softplus(sP(r, 0) * weights.getParameters()(0, i));
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