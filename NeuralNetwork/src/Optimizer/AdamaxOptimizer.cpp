#include "AdamaxOptimizer.h"

using namespace std;

AdamaxOptimizer::AdamaxOptimizer(vector<NeuralLayer*>* layers, int batchSize, double eta, double beta1, double beta2, double epsilon) : Optimizer(layers)
{
	this->batchSize = batchSize;
	this->eta = eta;
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->epsilon = epsilon;
	this->t = 0;
}

AdamaxOptimizer::AdamaxOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
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
	if (additionalParameters.find(BETA1) == additionalParameters.end())
	{
		this->beta1 = 0.9;
	}
	else
	{
		this->beta1 = additionalParameters[BETA1];
	}
	if (additionalParameters.find(BETA2) == additionalParameters.end())
	{
		this->beta2 = 0.999;
	}
	else
	{
		this->beta2 = additionalParameters[BETA2];
	}
	if (additionalParameters.find(EPSILON) == additionalParameters.end())
	{
		this->epsilon = 1e-7;
	}
	else
	{
		this->epsilon = additionalParameters[EPSILON];
	}
	this->t = 0;
}

void AdamaxOptimizer::setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient)
{
	auto g = applyRegularization(parameters, gradient);
	auto parameterID = parameters.getID();
	if (m.find(parameterID) == m.end() ||
		u.find(parameterID) == u.end())
	{
		m[parameterID] = xt::zeros<double>(g.shape());
		u[parameterID] = xt::zeros<double>(g.shape());
		t = 0;
	}
	else { }
	t++;
	m[parameterID] = (beta1 * m[parameterID]) + ((1 - beta1) * g);
	u[parameterID] = xt::maximum(beta2 * u[parameterID], xt::abs(g));
	auto mHat = m[parameterID] / (1 - std::pow(beta1, t));
	xt::xarray<double> optimizedGradient = -eta / (u[parameterID] + epsilon) * mHat;
	parameters.setDeltaParameters(optimizedGradient);
}