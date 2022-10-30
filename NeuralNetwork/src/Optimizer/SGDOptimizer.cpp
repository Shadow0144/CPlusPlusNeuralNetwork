#include "SGDOptimizer.h"

using namespace std;

SGDOptimizer::SGDOptimizer(vector<NeuralLayer*>* layers, int batchSize, double eta, double gamma, bool nesterov) : Optimizer(layers)
{
	this->batchSize = batchSize;
	this->eta = eta;
	this->gamma = gamma;
	this->nesterov = nesterov;
}

SGDOptimizer::SGDOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(BATCH_SIZE) == additionalParameters.end())
	{
		this->batchSize = -1;
	}
	else
	{
		this->batchSize = additionalParameters[BATCH_SIZE];
	}
	if (additionalParameters.find(ETA) == additionalParameters.end())
	{
		this->eta = 0.01;
	}
	else
	{
		this->eta = additionalParameters[ETA];
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

void SGDOptimizer::setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient)
{
	auto g = applyRegularization(parameters, gradient);
	auto parameterID = parameters.getID();
	xt::xarray<double> optimizedGradient;
	if (gamma > 0.0)
	{
		if (previousVelocity.find(parameterID) == previousVelocity.end())
		{
			previousVelocity[parameterID] = xt::zeros<double>(g.shape());
		}
		else { }
		xt::xarray<double> velocity = gamma * previousVelocity[parameterID] + eta * g;
		optimizedGradient = -velocity;
		previousVelocity[parameterID] = velocity;
	}
	else // Skip storing and updating the velocity if the momentum is 0
	{
		optimizedGradient = -eta * g; // Multiply by the learning rate
	}
	parameters.setDeltaParameters(optimizedGradient);
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