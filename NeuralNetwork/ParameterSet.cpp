#include "ParameterSet.h"
#include "NeuralNetwork.h"

#pragma warning(push, 0)
#include <iostream>
#include <time.h>
#include <xtensor/xrandom.hpp>
#include <mutex>  // For std::unique_lock
#pragma warning(pop)

using namespace std;

long ParameterSet::nextParameterID = 0;

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
	parameterID = nextParameterID++;
}

ParameterSet::ParameterSet(const ParameterSet& parameterSet)
{
	parameterSet.weightsMutex.lock_shared();
	this->setParameters(parameterSet.parameters);
	parameterSet.weightsMutex.unlock_shared();
	parameterID = nextParameterID++;
}

long ParameterSet::getID()
{
	return parameterID;
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
}

void ParameterSet::setParametersRandom(const std::vector<size_t>& numParameters)
{
	weightsMutex.lock();
	parameters = 2.0 * (xt::random::rand<double>(numParameters) - 0.5);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersPositiveRandom(size_t numParameters)
{
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = xt::random::rand<double>(pSize);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersPositiveRandom(const std::vector<size_t>& numParameters)
{
	weightsMutex.lock();
	parameters = xt::random::rand<double>(numParameters);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersZero(size_t numParameters)
{
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = xt::zeros<double>(pSize);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersZero(const std::vector<size_t>& numParameters)
{
	weightsMutex.lock();
	parameters = xt::zeros<double>(numParameters);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersOne(size_t numParameters)
{
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = xt::ones<double>(pSize);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersOne(const std::vector<size_t>& numParameters)
{
	weightsMutex.lock();
	parameters = xt::ones<double>(numParameters);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

xt::xarray<double> ParameterSet::getDeltaParameters() const
{
	return deltaParameters;
}

void ParameterSet::setDeltaParameters(const xt::xarray<double>& deltaParameters)
{
	this->deltaParameters = deltaParameters;
}

void ParameterSet::applyDeltaParameters()
{
	weightsMutex.lock();
	parameters += deltaParameters;
	weightsMutex.unlock();
	deltaParameters = xt::zeros<double>(parameters.shape());
}