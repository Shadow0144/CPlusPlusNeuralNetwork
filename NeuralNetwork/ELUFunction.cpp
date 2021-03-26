#include "ELUFunction.h"
#include "NeuralNetworkFileHelper.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xnpy.hpp>
#pragma warning(pop)

using namespace std;

const std::string ELUFunction::ALPHA = "eta"; // Parameter string [OPTIONAL]

ELUFunction::ELUFunction()
{

}

ELUFunction::ELUFunction(double alpha)
{
	this->alpha = alpha;
}

double ELUFunction::activate(double z) const
{
	return ELU(z);
}

double ELUFunction::ELU(double z) const
{
	return ((z < 0.0) ? (alpha * (exp(z) - 1.0)) : z);
}

xt::xarray<double> ELUFunction::ELU(const xt::xarray<double>& z) const
{
	auto mask = (z > 0.0);
	return ((1.0 - mask) * (alpha * (exp(z) - 1.0)) + (mask * z));
}

xt::xarray<double> ELUFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return ELU(inputs);
}

xt::xarray<double> ELUFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return (sigmas * xt::maximum(lastOutput + alpha, 1.0));
}

double ELUFunction::getParameter(const std::string& parameterName) const
{
	if (parameterName.compare(ELUFunction::ALPHA) == 0)
	{
		return alpha;
	}
	else
	{
		throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
	}
}

void ELUFunction::setParameter(const std::string& parameterName, double value)
{
	if (parameterName.compare(ELUFunction::ALPHA) == 0)
	{
		alpha = value;
	}
	else
	{
		throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
	}
}

void ELUFunction::saveParameters(std::string fileName)
{
	xt::dump_npy(fileName + "_alpha.npy", xt::xarray<double>({ alpha }));
}

void ELUFunction::loadParameters(std::string fileName)
{
	bool exists = NeuralNetworkFileHelper::fileExists(fileName + "_alpha.npy");
	if (exists)
	{
		alpha = xt::load_npy<double>(fileName + "_alpha.npy")(0);
	}
	else
	{
		cout << "Parameter file " + fileName + "_alpha.npy" + "not found" << endl;
	}
}

double ELUFunction::getAlpha() const
{
	return alpha;
}

void ELUFunction::setAlpha(double alpha)
{
	this->alpha = alpha;
}

void ELUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	ActivationFunction::approximateFunction(canvas, origin, scale, numUnits, weights);
}