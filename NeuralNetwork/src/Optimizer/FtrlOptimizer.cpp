#include "Optimizer/FtrlOptimizer.h"

using namespace std;

FtrlOptimizer::FtrlOptimizer(vector<NeuralLayer*>* layers, int batchSize, double alpha, double beta, double lambda1, double lambda2) : Optimizer(layers)
{
	this->batchSize = batchSize;
	this->alpha = alpha;
	this->beta = beta;
	this->lambda1 = lambda1;
	this->lambda2 = lambda2;
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
		this->lambda1 = 0.001;
	}
	else
	{
		this->lambda1 = additionalParameters[LAMDA1];
	}
	if (additionalParameters.find(LAMDA2) == additionalParameters.end())
	{
		this->lambda2 = 0.001;
	}
	else
	{
		this->lambda2 = additionalParameters[LAMDA2];
	}
}

void FtrlOptimizer::setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient)
{
	auto g = applyRegularization(parameters, gradient);
	auto parameterID = parameters.getID();
	if (z.find(parameterID) == z.end() ||
		n.find(parameterID) == n.end())
	{
		z[parameterID] = xt::zeros<double>(g.shape());
		n[parameterID] = xt::zeros<double>(g.shape());
	}
	else { }
	auto w = parameters.getParameters();
	auto g2 = xt::pow(g, 2.0);
	auto sigma = (xt::sqrt(n[parameterID] + g2) - xt::sqrt(n[parameterID])) / alpha;
	z[parameterID] = z[parameterID] + g - (sigma * w);
	n[parameterID] = n[parameterID] + g2;
	auto mask = xt::abs(z[parameterID]) > lambda1; // Set weights below lambda1 to zero
	auto newWeights = ((xt::sign(z[parameterID]) * lambda1) - z[parameterID]) / (((beta + xt::sqrt(n[parameterID])) / alpha) + lambda2);
	xt::xarray<double> optimizedGradient = (mask * newWeights); // Only weights above lambda1 will be non-zero
	optimizedGradient -= w; // FTRL sets the weights rather than adjusting them by a delta weight, so subtract the current weight
	parameters.setDeltaParameters(optimizedGradient); // Set the delta instead of the weights for testing for convergence reasons
}