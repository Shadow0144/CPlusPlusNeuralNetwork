#include "SGDOptimizer.h"

using namespace std;

const std::string SGDOptimizer::ETA = "eta"; // Parameter string [REQUIRED]
const std::string SGDOptimizer::BATCH_SIZE = "batchSize"; // Parameter string [OPTIONAL]
const std::string SGDOptimizer::GAMMA = "gamma"; // Parameter string [OPTIONAL]
const std::string SGDOptimizer::NESTEROV = "nesterov"; // Parameter string [OPTIONAL]

SGDOptimizer::SGDOptimizer(vector<NeuralLayer*>* layers, double eta, int batchSize, double gamma, bool nesterov) : Optimizer(layers)
{
	this->eta = eta;
	this->batchSize = batchSize;
	this->gamma = gamma;
	this->nesterov = nesterov;
}

SGDOptimizer::SGDOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(ETA) == additionalParameters.end())
	{
		throw std::invalid_argument(std::string("Missing required parameter: ") +
			"SGDOptimizer::ETA" + " (\"" + ETA + "\")");
	}
	else
	{
		this->eta = additionalParameters[ETA];
		if (additionalParameters.find(BATCH_SIZE) == additionalParameters.end())
		{
			this->batchSize = -1;
		}
		else
		{
			this->batchSize = additionalParameters[BATCH_SIZE];
		}
		if (additionalParameters.find(GAMMA) == additionalParameters.end())
		{
			this->gamma = 0;
		}
		else
		{
			this->gamma = additionalParameters[GAMMA];
		}
		if (additionalParameters.find(NESTEROV) == additionalParameters.end())
		{
			this->nesterov = false;
		}
		else
		{
			this->nesterov = (additionalParameters[NESTEROV] != 0);
		}
	}
}

double SGDOptimizer::backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
{
	bool converged = true;

	// We need to limit the size of the data being processed at once for memory reasons, but the update will still be on the entire batch
	const int N = inputs.shape()[0];
	const int INTERNAL_BATCHES = ceil(N / INTERNAL_BATCH_LIMIT);
	int iBatchSize = N / INTERNAL_BATCHES;

	size_t layerCount = layers->size();
	auto shape = layers->at(layerCount - 1)->getOutputShape();
	shape[0] = N;
	if (nesterov) // If Nesterov accelerated gradient (NAG) is enabled, we want to find the gradient of the predicted weights
	{
		substituteAllParameters();
	}
	else { }
	xt::xarray<double> predicted = xt::xarray<double>(shape);

	for (int i = 0; i < INTERNAL_BATCHES; i++)
	{
		// Set up the batch
		int iBatchStart = ((i + 0) * iBatchSize) % N;
		int iBatchEnd = ((i + 1) * iBatchSize) % N;
		if ((iBatchEnd - iBatchStart) != iBatchSize)
		{
			iBatchEnd = N;
		}
		else { }

		xt::xstrided_slice_vector iBatchSV({ xt::range(iBatchStart, iBatchEnd), xt::ellipsis() });

		// Feed forward and calculate the gradient
		xt::xarray<double> predicted = feedForwardTrain(xt::strided_view(inputs, iBatchSV));
		xt::xarray<double> sigma = errorFunction->getGradient(predicted, xt::strided_view(targets, iBatchSV));

		// Backpropagate through the layers until the input layer
		for (int l = (layerCount - 1); l > 0; l--)
		{
			sigma = layers->at(l)->getGradient(sigma, this);
		}
	}

	if (nesterov) // If Nesterov accelerated gradient (NAG) is enabled, restore the weights before updating them
	{
		substituteAllParameters();
	}
	else { }

	// Apply the backpropagation
	double deltaSum = 0.0;
	for (int l = 0; l < layerCount; l++)
	{
		deltaSum += layers->at(l)->applyBackPropagate();
	}

	return deltaSum;
}

xt::xarray<double> SGDOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	xt::xarray<double> optimizedGradient;
	if (gamma > 0.0)
	{
		if (previousVelocity.find(parameterID) == previousVelocity.end())
		{
			previousVelocity[parameterID] = xt::zeros<double>(gradient.shape());
		}
		else { }
		xt::xarray<double> velocity = gamma * previousVelocity[parameterID] + eta * gradient;
		optimizedGradient = -velocity;
		previousVelocity[parameterID] = velocity;
	}
	else // Skip storing and updating the velocity if the momentum is 0
	{
		optimizedGradient = -eta * gradient; // Multiply by the learning rate
	}
	return optimizedGradient;
}

void SGDOptimizer::substituteParameters(ParameterSet& parameterSet)
{
	auto parameters = parameterSet.getParameters();
	long parameterID = parameterSet.getID();
	if (previousVelocity.find(parameterID) == previousVelocity.end())
	{
		previousVelocity[parameterID] = xt::zeros<double>(parameters.shape());
	}
	else { }
	parameterSet.setParameters(parameterSet.getParameters() - (gamma * previousVelocity[parameterID]));
}

void SGDOptimizer::restoreParameters(ParameterSet& parameterSet)
{
	long parameterID = parameterSet.getID();
	parameterSet.setParameters(parameterSet.getParameters() + (gamma * previousVelocity[parameterID]));
}