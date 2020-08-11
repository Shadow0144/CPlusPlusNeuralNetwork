#include "ParameterSet.h"
#include "NeuralNetwork.h"

#pragma warning(push, 0)
#include <iostream>
#include <time.h>
#include <xtensor/xrandom.hpp>
#pragma warning(pop)

using namespace std;


static bool seedSet; // Defaults to false
ParameterSet::ParameterSet()
{
	if (!seedSet)
	{
		xt::random::seed(time(NULL));
		seedSet = true;
	}
	else { }
	parameters = xt::xarray<double>();
	deltaParameters = xt::xarray<double>();
	batchSize = 0;
}

xt::xarray<double> ParameterSet::getParameters()
{ 
	return parameters;
}

void ParameterSet::setParametersRandom(size_t numParameters)
{
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);
	parameters = 2.0 * (xt::random::rand<double>(pSize) - 0.5);
	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersRandom(std::vector<size_t> numParameters)
{
	parameters = 2.0 * (xt::random::rand<double>(numParameters) - 0.5);
	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersZero(size_t numParameters)
{
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);
	parameters = xt::zeros<double>(pSize);
	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersZero(std::vector<size_t> numParameters)
{
	parameters = xt::zeros<double>(numParameters);
	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersOne(size_t numParameters)
{
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);
	parameters = xt::ones<double>(pSize);
	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersOne(std::vector<size_t> numParameters)
{
	parameters = xt::ones<double>(numParameters);
	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

xt::xarray<double> ParameterSet::getDeltaParameters()
{
	return deltaParameters;
}

void ParameterSet::incrementDeltaParameters(xt::xarray<double> deltaParameters)
{
	this->deltaParameters += deltaParameters;
	batchSize++;
}

void ParameterSet::applyDeltaParameters()
{
	parameters += (deltaParameters / batchSize);
	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}