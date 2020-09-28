#include "SigmoidFunction.h"
#include "NeuralLayer.h"

SigmoidFunction::SigmoidFunction(size_t incomingUnits, size_t numUnits)
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

double SigmoidFunction::activate(double z)
{
	return sigmoid(z);
}

double SigmoidFunction::sigmoid(double z)
{
	return (1.0 / (1.0 + exp(-z)));
}
	
xt::xarray<double> SigmoidFunction::sigmoid(xt::xarray<double> z)
{
	return (1.0 / (1.0 + exp(-z)));
}

xt::xarray<double> SigmoidFunction::feedForward(xt::xarray<double> inputs)
{
	auto dotProductResult = dotProduct(inputs);
	lastOutput = sigmoid(dotProductResult);
	return lastOutput;
}

xt::xarray<double> SigmoidFunction::backPropagate(xt::xarray<double> sigmas)
{
	return denseBackpropagate(sigmas * activationDerivative());
}

xt::xarray<double> SigmoidFunction::activationDerivative()
{
	return lastOutput * (1.0 - lastOutput);
}

void SigmoidFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	Function::approximateFunction(canvas, origin, scale);
}