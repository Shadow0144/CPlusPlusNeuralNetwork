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

void SGDOptimizer::backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
{
	if (nesterov) // If Nesterov accelerated gradient (NAG) is enabled, we want to find the gradient of the predicted weights
	{
		substituteAllParameters();
	}
	else { }

	Optimizer::backPropagateBatch(inputs, targets);
	
	if (nesterov) // If Nesterov accelerated gradient (NAG) is enabled, restore the weights before updating them
	{
		restoreAllParameters();
	}
	else { }
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