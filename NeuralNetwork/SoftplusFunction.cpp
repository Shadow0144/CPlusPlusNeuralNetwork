#include "SoftplusFunction.h"
#include "NeuralLayer.h"
#include "NeuralNetworkFileHelper.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xnpy.hpp>
#pragma warning(pop)

using namespace std;

const std::string SoftplusFunction::K = "k"; // Parameter string [OPTIONAL]

SoftplusFunction::SoftplusFunction()
{

}

SoftplusFunction::SoftplusFunction(double k)
{
	this->k = k;
}

double SoftplusFunction::activate(double z) const
{
	return softplus(z);
}

double SoftplusFunction::softplus(double z) const
{
	return (log(1.0 + exp(k * z)) / k);
}

xt::xarray<double> SoftplusFunction::softplus(const xt::xarray<double>& z) const
{
	return (log(1.0 + exp(k * z)) / k);
}

xt::xarray<double> SoftplusFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return softplus(inputs);
}

xt::xarray<double> SoftplusFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	// TODO!!!
	//return (1.0 / (1.0 + exp(-k * xt::linalg::tensordot(lastInput, weights.getParameters(), 1))));
	return lastInput;
}

double SoftplusFunction::getParameter(const std::string& parameterName) const
{
	if (parameterName.compare(SoftplusFunction::K) == 0)
	{
		return k;
	}
	else
	{
		throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
	}
}

void SoftplusFunction::setParameter(const std::string& parameterName, double value)
{
	if (parameterName.compare(SoftplusFunction::K) == 0)
	{
		k = value;
	}
	else
	{
		throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
	}
}

void SoftplusFunction::saveParameters(std::string fileName)
{
	xt::dump_npy(fileName + "_k.npy", xt::xarray<double>({ k }));
}

void SoftplusFunction::loadParameters(std::string fileName)
{
	bool exists = NeuralNetworkFileHelper::fileExists(fileName + "_k.npy");
	if (exists)
	{
		k = xt::load_npy<double>(fileName + "_k.npy")(0);
	}
	else
	{
		cout << "Parameter file " + fileName + "_k.npy" + "not found" << endl;
	}
}

double SoftplusFunction::getK() const
{
	return k;
}

void SoftplusFunction::setK(double k)
{
	this->k = k;
}

void SoftplusFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}