#include "RPropOptimizer.h"

using namespace std;

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

void RPropOptimizer::setDeltaWeight(ParameterSet& parameters, const xt::xarray<double>& gradient)
{
	auto g = applyRegularization(parameters, gradient);
	auto parameterID = parameters.getID();
	if (alpha.find(parameterID) == alpha.end() ||
		prevG.find(parameterID) == prevG.end())
	{
		alpha[parameterID] = xt::ones<double>(g.shape());
		prevG[parameterID] = xt::zeros<double>(g.shape());
	}
	else { }
	auto signs = prevG[parameterID] * g;
	alpha[parameterID] *= ((signs < 0) * shrinkAlpha) + ((signs < 0) * growAlpha) + ((xt::equal(signs, 0)) * 1);
	alpha[parameterID] = xt::maximum(xt::minimum(alpha[parameterID], maxAlpha), minAlpha);
	prevG[parameterID] = g;
	xt::xarray<double> optimizedGradient = -alpha[parameterID] * xt::sign(g);
	parameters.setDeltaParameters(optimizedGradient);
}