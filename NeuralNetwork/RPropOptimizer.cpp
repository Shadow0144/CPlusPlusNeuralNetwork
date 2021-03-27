#include "RPropOptimizer.h"

using namespace std;

const std::string RPropOptimizer::MIN_ALPHA = "minAlpha"; // Parameter string [OPTIONAL]
const std::string RPropOptimizer::MAX_ALPHA = "maxAlpha"; // Parameter string [OPTIONAL]
const std::string RPropOptimizer::SHRINK_ALPHA = "shrinkAlpha"; // Parameter string [OPTIONAL]
const std::string RPropOptimizer::GROW_ALPHA = "growAlpha"; // Parameter string [OPTIONAL]

RPropOptimizer::RPropOptimizer(vector<NeuralLayer*>* layers, double shrinkAlpha, double growAlpha, double minAlpha, double maxAlpha) : Optimizer(layers)
{
	this->shrinkAlpha = shrinkAlpha;
	this->growAlpha = growAlpha;
	this->minAlpha = minAlpha;
	this->maxAlpha = maxAlpha;
}

RPropOptimizer::RPropOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(SHRINK_ALPHA) == additionalParameters.end())
	{
		this->shrinkAlpha = shrinkAlpha;
	}
	else
	{
		this->shrinkAlpha = additionalParameters[SHRINK_ALPHA];
	}
	if (additionalParameters.find(GROW_ALPHA) == additionalParameters.end())
	{
		this->growAlpha = 1.2;
	}
	else
	{
		this->growAlpha = additionalParameters[GROW_ALPHA];
	}
	if (additionalParameters.find(MIN_ALPHA) == additionalParameters.end())
	{
		this->minAlpha = 0.0001;
	}
	else
	{
		this->minAlpha = additionalParameters[MIN_ALPHA];
	}
	if (additionalParameters.find(MAX_ALPHA) == additionalParameters.end())
	{
		this->maxAlpha = 50.0;
	}
	else
	{
		this->maxAlpha = additionalParameters[MAX_ALPHA];
	}
}

xt::xarray<double> RPropOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	if (alpha.find(parameterID) == alpha.end() ||
		g.find(parameterID) == g.end())
	{
		alpha[parameterID] = xt::ones<double>(gradient.shape());
		g[parameterID] = xt::zeros<double>(gradient.shape());
	}
	else { }
	auto signs = g[parameterID] * gradient;
	alpha[parameterID] *= ((signs < 0) * shrinkAlpha) + ((signs < 0) * growAlpha) + ((xt::equal(signs, 0)) * 1);
	alpha[parameterID] = xt::maximum(xt::minimum(alpha[parameterID], maxAlpha), minAlpha);
	g[parameterID] = gradient;
	xt::xarray<double> optimizedGradient = -alpha[parameterID] * xt::sign(gradient);
	return optimizedGradient;
}