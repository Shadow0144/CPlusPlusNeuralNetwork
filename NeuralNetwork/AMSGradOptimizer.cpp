#include "AMSGradOptimizer.h"

using namespace std;

const std::string AMSGradOptimizer::ETA = "eta"; // Parameter string [REQUIRED]
const std::string AMSGradOptimizer::BATCH_SIZE = "batchSize"; // Parameter string [OPTIONAL]
const std::string AMSGradOptimizer::BETA1 = "beta1"; // Parameter string [OPTIONAL]
const std::string AMSGradOptimizer::BETA2 = "beta2"; // Parameter string [OPTIONAL]
const std::string AMSGradOptimizer::EPSILON = "epsilon"; // Parameter string [OPTIONAL]

AMSGradOptimizer::AMSGradOptimizer(vector<NeuralLayer*>* layers, double eta, int batchSize, double beta1, double beta2, double epsilon) : Optimizer(layers)
{
	this->eta = eta;
	this->batchSize = batchSize;
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->epsilon = epsilon;
	this->t = 0;
}

AMSGradOptimizer::AMSGradOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(ETA) == additionalParameters.end())
	{
		throw std::invalid_argument(std::string("Missing required parameter: ") +
			"AMSGradOptimizer::ETA" + " (\"" + ETA + "\")");
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

xt::xarray<double> AMSGradOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	if (m.find(parameterID) == m.end() ||
		v.find(parameterID) == v.end())
	{
		m[parameterID] = xt::zeros<double>(gradient.shape());
		v[parameterID] = xt::zeros<double>(gradient.shape());
		vHat[parameterID] = xt::zeros<double>(gradient.shape());
		t = 0;
	}
	else { }
	t++;
	m[parameterID] = (beta1 * m[parameterID]) + ((1 - beta1) * gradient);
	v[parameterID] = (beta2 * v[parameterID]) + ((1 - beta2) * xt::pow(gradient, 2.0));
	vHat[parameterID] = xt::maximum(vHat[parameterID], v[parameterID]);
	xt::xarray<double> optimizedGradient = -eta / (xt::pow(vHat[parameterID], 0.5) + epsilon) * m[parameterID];
	return optimizedGradient;
}