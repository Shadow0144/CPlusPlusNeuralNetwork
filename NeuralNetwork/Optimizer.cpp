#include "Optimizer.h"

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