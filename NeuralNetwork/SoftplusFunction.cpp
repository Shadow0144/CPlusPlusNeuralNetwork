#include "SoftplusFunction.h"

#include <iostream>

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
	return (1.0 / (1.0 + exp(-k * lastInput * weights.getParameters())));
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
	/*Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	int r = 3;
	double range = 3.0;

	int resolution = (r * 4) + 1;
	MatrixXd sP(resolution, 2);
	for (int r = 0; r < resolution; r++)
	{
		sP(r, 0) = range * (2.0 * r) / (resolution - 1.0) - range;
		sP(r, 1) = softplus(sP(r, 0), k);
	}

	float rescale = (1.0 / range) * DRAW_LEN * scale;
	for (int d = 0; d < (resolution - 3); d += 3)
	{
		MatrixXd points = approximateBezier(sP.block(d, 0, 4, 2));
		canvas->AddBezierCurve(
			ImVec2(origin.x + (points(0, 0) * rescale), origin.y - (points(0, 1) * rescale)),
			ImVec2(origin.x + (points(1, 0) * rescale), origin.y - (points(1, 1) * rescale)),
			ImVec2(origin.x + (points(2, 0) * rescale), origin.y - (points(2, 1) * rescale)),
			ImVec2(origin.x + (points(3, 0) * rescale), origin.y - (points(3, 1) * rescale)),
			BLACK, 1);
	}*/
}