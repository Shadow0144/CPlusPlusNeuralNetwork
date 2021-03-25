#include "Optimizer.h"

#include "NeuralLayer.h"

using namespace std;

Optimizer::Optimizer(vector<NeuralLayer*>* layers)
{
	this->layers = layers;
	this->errorFunction = nullptr;
}

Optimizer::~Optimizer()
{

}

void Optimizer::setErrorFunction(ErrorFunction* errorFunction)
{
	this->errorFunction = errorFunction;
}

// Called internally only, internal batching is handled in getDeltaWeight
xt::xarray<double> Optimizer::feedForwardTrain(const xt::xarray<double>& inputs)
{
	xt::xarray<double> predicted = inputs;

	size_t layerCount = layers->size();
	for (int i = 0; i < layerCount; i++) // Loop through the layers
	{
		predicted = layers->at(i)->feedForwardTrain(predicted);
	}

	return predicted;
}

void Optimizer::substituteParameters(ParameterSet& parameterSet)
{
	// Do nothing
}

void Optimizer::restoreParameters(ParameterSet& parameterSet)
{
	// Do nothing
}

void Optimizer::substituteAllParameters()
{
	const int L = layers->size();
	for (int i = 0; i < L; i++)
	{
		layers->at(i)->substituteParameters(this);
	}
}

void Optimizer::restoreAllParameters()
{
	const int L = layers->size();
	for (int i = 0; i < L; i++)
	{
		layers->at(i)->restoreParameters(this);
	}
}