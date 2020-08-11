#include "LeakyReLUFunction.h"

#include <iostream>

using namespace std;

LeakyReLUFunction::LeakyReLUFunction(size_t incomingUnits, size_t numUnits)
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

xt::xarray<double> LeakyReLUFunction::leakyReLU(xt::xarray<double> z)
{
	return xt::maximum(a * z, z);
}

xt::xarray<double> LeakyReLUFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	return leakyReLU(dotProductResult);
}

xt::xarray<double> LeakyReLUFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> LeakyReLUFunction::activationDerivative()
{
	return ((lastOutput(0) > 0.0) ? 1.0 : a); // TODO
}

double LeakyReLUFunction::getA() 
{ 
	return a;
}

void LeakyReLUFunction::setA(double a) 
{
	this->a = a;
}

void LeakyReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	double slope = weights.getParameters()(0);
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

	ImVec2 l_start(origin.x + (DRAW_LEN * x1 * scale), origin.y - (DRAW_LEN * y1 * scale));
	ImVec2 l_mid(origin.x, origin.y);
	ImVec2 l_end(origin.x + (DRAW_LEN * x2 * scale), origin.y - (DRAW_LEN * y2 * scale));

	canvas->AddLine(l_start, l_mid, BLACK);
	canvas->AddLine(l_mid, l_end, BLACK);
}