#include "FtrlOptimizer.h"

using namespace std;

FtrlOptimizer::FtrlOptimizer(vector<NeuralLayer*>* layers, int batchSize, double alpha, double beta, double lamda1, double lamda2) : Optimizer(layers)
{
	this->batchSize = batchSize;
	this->alpha = alpha;
	this->beta = beta;
	this->lamda1 = lamda1;
	this->lamda2 = lamda2;
}

FtrlOptimizer::FtrlOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(BATCH_SIZE) == additionalParameters.end())
	{
		this->batchSize = -1;
	}
	else
	{
		this->batchSize = additionalParameters[BATCH_SIZE];
	}
	if (additionalParameters.find(ALPHA) == additionalParameters.end())
	{
		this->alpha = 1.0;
	}
	else
	{
		this->alpha = additionalParameters[ALPHA];
	}
	if (additionalParameters.find(BETA) == additionalParameters.end())
	{
		this->beta = 1.0;
	}
	else
	{
		this->beta = additionalParameters[BETA];
	}
	if (additionalParameters.find(LAMDA1) == additionalParameters.end())
	{
		this->lamda1 = 0.001;
	}
	else
	{
		this->lamda1 = additionalParameters[LAMDA1];
	}
	if (additionalParameters.find(LAMDA2) == additionalParameters.end())
	{
		this->lamda2 = 0.001;
	}
	else
	{
		this->lamda2 = additionalParameters[LAMDA2];
	}
}

double FtrlOptimizer::backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
{
	substituteAllParameters(); // Need to update w
	return Optimizer::backPropagate(inputs, targets);
}

xt::xarray<double> FtrlOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	if (z.find(parameterID) == z.end() ||
		n.find(parameterID) == n.end())
	{
		z[parameterID] = xt::zeros<double>(gradient.shape());
		n[parameterID] = xt::zeros<double>(gradient.shape());
	}
	else { }
	auto g2 = xt::pow(gradient, 2.0);
	auto sigma = (xt::sqrt(n[parameterID] + g2) - xt::sqrt(n[parameterID])) / alpha;
	z[parameterID] = z[parameterID] + gradient - (sigma * w[parameterID]);
	n[parameterID] = n[parameterID] + g2;
	auto mask = xt::abs(z[parameterID]) > lamda1; // Set weights below lamda1 to zero
	auto newWeights = ((xt::sign(z[parameterID]) * lamda1) - z[parameterID]) / (((beta + xt::sqrt(n[parameterID])) / alpha) + lamda2);
	xt::xarray<double> optimizedGradient = (mask * newWeights); // Only weights above lamda1 will be non-zero
	optimizedGradient -= w[parameterID]; // FTRL sets the weights rather than adjusting them by a delta weight, so subtract the current weight
	return optimizedGradient;
}

void FtrlOptimizer::substituteParameters(ParameterSet& parameterSet)
{
	w[parameterSet.getID()] = parameterSet.getParameters();
}