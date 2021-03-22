#include "ParameterSet.h"
#include "NeuralNetwork.h"

#pragma warning(push, 0)
#include <iostream>
#include <time.h>
#include <xtensor/xrandom.hpp>
#include <mutex>  // For std::unique_lock
#pragma warning(pop)

using namespace std;

static bool seedSet; // Defaults to false
ParameterSet::ParameterSet()
{
	if (!seedSet)
	{
		xt::random::seed(time(nullptr));
		seedSet = true;
	}
	else { }
	weightsMutex.lock();
	parameters = xt::xarray<double>();
	weightsMutex.unlock();
	deltaParameters = xt::xarray<double>();
	batchSize = 0;
}

ParameterSet::ParameterSet(const ParameterSet& parameterSet)
{
	parameterSet.weightsMutex.lock_shared();
	this->setParameters(parameterSet.parameters);
	parameterSet.weightsMutex.unlock_shared();
}

xt::xarray<double> ParameterSet::getParameters() const
{ 
	weightsMutex.lock_shared();
	const xt::xarray<double> rParameters = xt::xarray<double>(parameters);
	weightsMutex.unlock_shared();
	return rParameters;
}

void ParameterSet::setParameters(const xt::xarray<double>& parameters)
{
	weightsMutex.lock();
	this->parameters = parameters;
	weightsMutex.unlock();
}

void ParameterSet::setParametersRandom(size_t numParameters)
{
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = 2.0 * (xt::random::rand<double>(pSize) - 0.5);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersRandom(const std::vector<size_t>& numParameters)
{
	weightsMutex.lock();
	parameters = 2.0 * (xt::random::rand<double>(numParameters) - 0.5);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersZero(size_t numParameters)
{
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = xt::zeros<double>(pSize);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersZero(const std::vector<size_t>& numParameters)
{
	weightsMutex.lock();
	parameters = xt::zeros<double>(numParameters);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersOne(size_t numParameters)
{
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = xt::ones<double>(pSize);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

void ParameterSet::setParametersOne(const std::vector<size_t>& numParameters)
{
	weightsMutex.lock();
	parameters = xt::ones<double>(numParameters);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}

xt::xarray<double> ParameterSet::getDeltaParameters() const
{
	return deltaParameters;
}

void ParameterSet::incrementDeltaParameters(const xt::xarray<double>& deltaParameters)
{
	this->deltaParameters += deltaParameters;
	batchSize++;
}

void ParameterSet::applyDeltaParameters()
{
	weightsMutex.lock();
	if (batchSize != 0)
	{
		parameters += (deltaParameters / batchSize);
	}
	else { }
	weightsMutex.unlock();
	deltaParameters = xt::zeros<double>(parameters.shape());
	batchSize = 0;
}