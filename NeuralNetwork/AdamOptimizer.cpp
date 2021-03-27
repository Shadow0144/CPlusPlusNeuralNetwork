#include "AdamOptimizer.h"

const double INTERNAL_BATCH_LIMIT = 20;

using namespace std;

const std::string AdamOptimizer::ETA = "eta"; // Parameter string [REQUIRED]
const std::string AdamOptimizer::BATCH_SIZE = "batchSize"; // Parameter string [OPTIONAL]
const std::string AdamOptimizer::BETA1 = "beta1"; // Parameter string [OPTIONAL]
const std::string AdamOptimizer::BETA2 = "beta2"; // Parameter string [OPTIONAL]
const std::string AdamOptimizer::EPSILON = "epsilon"; // Parameter string [OPTIONAL]

AdamOptimizer::AdamOptimizer(vector<NeuralLayer*>* layers, double eta, int batchSize, double beta1, double beta2, double epsilon) : Optimizer(layers)
{
	this->eta = eta;
	this->batchSize = batchSize;
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->epsilon = epsilon;
	this->t = 0;
}

AdamOptimizer::AdamOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
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
			this->epsilon = 1e-8;
		}
		else
		{
			this->epsilon = additionalParameters[EPSILON];
		}
		this->t = 0;
	}
}

xt::xarray<double> AdamOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	if (m.find(parameterID) == m.end() ||
		v.find(parameterID) == v.end())
	{
		m[parameterID] = xt::zeros<double>(gradient.shape());
		v[parameterID] = xt::zeros<double>(gradient.shape());
		t = 0;
	}
	else { }
	t++;
	m[parameterID] = (beta1 * m[parameterID]) + ((1 - beta1) * gradient);
	v[parameterID] = (beta2 * v[parameterID]) + ((1 - beta2) * xt::pow(gradient, 2.0));
	auto mHat = m[parameterID] / (1 - std::pow(beta1, t));
	auto vHat = v[parameterID] / (1 - std::pow(beta2, t));
	xt::xarray<double> optimizedGradient = -eta / (xt::pow(vHat, 0.5) + epsilon) * mHat;
	return optimizedGradient;
}