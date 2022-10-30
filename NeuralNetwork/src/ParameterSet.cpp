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
	unregularized = false;
}

ParameterSet::ParameterSet(const ParameterSet& parameterSet)
{
	parameterSet.weightsMutex.lock_shared();
	this->setParameters(parameterSet.parameters);
	parameterSet.weightsMutex.unlock_shared();
	this->hasBias = parameterSet.hasBias;
	this->unregularized = parameterSet.unregularized;
	parameterID = nextParameterID++;
}

long ParameterSet::getID()
{
	return parameterID;
}

xt::xarray<double> ParameterSet::getParameters() const
{ 
	weightsMutex.lock_shared();
	xt::xarray<double> rParameters = xt::xarray<double>(parameters);
	weightsMutex.unlock_shared();
	return rParameters;
}

xt::xarray<double> ParameterSet::getParametersWithoutBias() const
{
	weightsMutex.lock_shared();
	xt::xarray<double> rParameters = xt::view(parameters, xt::range(0, parameters.shape()[0] - 1), xt::all());
	weightsMutex.unlock_shared();
	return rParameters;
}

void ParameterSet::setParameters(const xt::xarray<double>& parameters, bool hasBias)
{
	this->hasBias = hasBias;
	weightsMutex.lock();
	this->parameters = parameters;
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersRandom(size_t numParameters, bool hasBias)
{
	this->hasBias = hasBias;
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = 2.0 * (xt::random::rand<double>(pSize) - 0.5);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersRandom(const std::vector<size_t>& numParameters, bool hasBias)
{
	this->hasBias = hasBias;
	weightsMutex.lock();
	parameters = 2.0 * (xt::random::rand<double>(numParameters) - 0.5);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersPositiveRandom(size_t numParameters, bool hasBias)
{
	this->hasBias = hasBias;
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = xt::random::rand<double>(pSize);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersPositiveRandom(const std::vector<size_t>& numParameters, bool hasBias)
{
	this->hasBias = hasBias;
	weightsMutex.lock();
	parameters = xt::random::rand<double>(numParameters);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersZero(size_t numParameters, bool hasBias)
{
	this->hasBias = hasBias;
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = xt::zeros<double>(pSize);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersZero(const std::vector<size_t>& numParameters, bool hasBias)
{
	this->hasBias = hasBias;
	weightsMutex.lock();
	parameters = xt::zeros<double>(numParameters);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersOne(size_t numParameters, bool hasBias)
{
	this->hasBias = hasBias;
	std::vector<size_t> pSize;
	pSize.push_back(numParameters);

	weightsMutex.lock();
	parameters = xt::ones<double>(pSize);
	weightsMutex.unlock();

	deltaParameters = xt::zeros<double>(parameters.shape());
}

void ParameterSet::setParametersOne(const std::vector<size_t>& numParameters, bool hasBias)
{
	this->hasBias = hasBias;
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

void ParameterSet::setUnregularized()
{
	unregularized = true;
}

bool ParameterSet::getUnregularized() const
{
	return unregularized;
}

double ParameterSet::getRegularizationLoss(double lambda1, double lambda2) const
{
	double loss = 0.0;
	if (!unregularized)
	{
		if (hasBias) // Bias is generally not regularized
		{
			auto parametersWithoutBias = getParametersWithoutBias();
			loss = (lambda1 * xt::sum(xt::abs(parametersWithoutBias))()) + (0.5 * (lambda2 * xt::sum(xt::pow(parametersWithoutBias, 2.0))()));
		}
		else
		{
			loss = (lambda1 * xt::sum(xt::abs(parameters))()) + (0.5 * (lambda2 * xt::sum(xt::pow(parameters, 2.0))()));
		}
	}
	else
	{
		// Do nothing
	}
	return loss;
}

xt::xarray<double> ParameterSet::getRegularizedGradient(double lambda1, double lambda2) const
{
	// Save calling this method by checking if we should regularize before calling this
	weightsMutex.lock_shared();
	xt::xarray<double> rGradient = xt::zeros<double>(parameters.shape());
	if (lambda1 != 0.0)
	{
		double halfLamda = 0.5 * lambda1;
		rGradient += lambda1 * ((parameters > 0.0) - (parameters < 0.0));
	}
	else { }
	if (lambda2 != 0.0)
	{
		rGradient += lambda2 * parameters;
	}
	else { }
	weightsMutex.unlock_shared();
	if (hasBias) // The bias terms are generally unregularized
	{
		xt::view(rGradient, parameters.shape()[0]-1, xt::all()) = 0.0;
	}
	else { }
	return rGradient;
}