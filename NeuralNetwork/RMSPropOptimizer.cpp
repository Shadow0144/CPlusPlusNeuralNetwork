#include "RMSPropOptimizer.h"

using namespace std;

RMSPropOptimizer::RMSPropOptimizer(vector<NeuralLayer*>* layers, int batchSize, double eta, double epsilon) : Optimizer(layers)
{
	this->batchSize = batchSize;
	this->eta = eta;
	this->epsilon = epsilon;
}

RMSPropOptimizer::RMSPropOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
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
	if (additionalParameters.find(EPSILON) == additionalParameters.end())
	{
		this->epsilon = 1e-7;
	}
	else
	{
		this->epsilon = additionalParameters[EPSILON];
	}
}

void RMSPropOptimizer::setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient)
{
	auto g = applyRegularization(parameters, gradient);
	auto parameterID = parameters.getID();
	if (Eg2.find(parameterID) == Eg2.end())
	{
		Eg2[parameterID] = xt::zeros<double>(g.shape());
	}
	else { }
	Eg2[parameterID] = (0.9 * Eg2[parameterID]) + ((0.1) * xt::pow(g, 2.0));
	xt::xarray<double> optimizedGradient = -eta * xt::pow(Eg2[parameterID] + epsilon, 0.5) * g;
	parameters.setDeltaParameters(optimizedGradient);
}