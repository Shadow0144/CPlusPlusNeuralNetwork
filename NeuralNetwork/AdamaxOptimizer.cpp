#include "AdamaxOptimizer.h"

using namespace std;

const std::string AdamaxOptimizer::ETA = "eta"; // Parameter string [REQUIRED]
const std::string AdamaxOptimizer::BATCH_SIZE = "batchSize"; // Parameter string [OPTIONAL]
const std::string AdamaxOptimizer::BETA1 = "beta1"; // Parameter string [OPTIONAL]
const std::string AdamaxOptimizer::BETA2 = "beta2"; // Parameter string [OPTIONAL]
const std::string AdamaxOptimizer::EPSILON = "epsilon"; // Parameter string [OPTIONAL]

AdamaxOptimizer::AdamaxOptimizer(vector<NeuralLayer*>* layers, double eta, int batchSize, double beta1, double beta2, double epsilon) : Optimizer(layers)
{
	this->eta = eta;
	this->batchSize = batchSize;
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->epsilon = epsilon;
	this->t = 0;
}

AdamaxOptimizer::AdamaxOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(ETA) == additionalParameters.end())
	{
		throw std::invalid_argument(std::string("Missing required parameter: ") +
			"AdamaxOptimizer::ETA" + " (\"" + ETA + "\")");
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
}

xt::xarray<double> AdamaxOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	if (m.find(parameterID) == m.end() ||
		u.find(parameterID) == u.end())
	{
		m[parameterID] = xt::zeros<double>(gradient.shape());
		u[parameterID] = xt::zeros<double>(gradient.shape());
		t = 0;
	}
	else { }
	t++;
	m[parameterID] = (beta1 * m[parameterID]) + ((1 - beta1) * gradient);
	u[parameterID] = xt::maximum(beta2 * u[parameterID], xt::abs(gradient));
	auto mHat = m[parameterID] / (1 - std::pow(beta1, t));
	xt::xarray<double> optimizedGradient = -eta / (u[parameterID] + epsilon) * mHat;
	return optimizedGradient;
}