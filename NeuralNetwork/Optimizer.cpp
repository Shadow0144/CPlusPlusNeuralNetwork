#include "Optimizer.h"

using namespace std;

Optimizer::Optimizer(vector<NeuralLayer*>* layers)
{
	this->layers = layers;

	this->currentEpoch = 0;

	this->errorFunction = nullptr;
	this->maxEpochs = -1;
	this->minError = -1;
	this->errorConvergenceThreshold = -1;
	this->weightConvergenceThreshold = -1;
	this->stoppingConditionFlags = new bool[4]; // There are four stopping conditions
	for (int i = 0; i < 4; i++)
	{
		stoppingConditionFlags[i] = false;
	}
}

Optimizer::~Optimizer()
{
	delete[] stoppingConditionFlags;
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