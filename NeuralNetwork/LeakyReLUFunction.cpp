#include "LeakyReLUFunction.h"
#include "NeuralLayer.h"
#include "NeuralNetworkFileHelper.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xnpy.hpp>
#pragma warning(pop)

using namespace std;

const std::string LeakyReLUFunction::A = "a"; // Parameter string [OPTIONAL]

LeakyReLUFunction::LeakyReLUFunction()
{

}

LeakyReLUFunction::LeakyReLUFunction(double a)
{
	this->a = a;
}

xt::xarray<double> LeakyReLUFunction::leakyReLU(const xt::xarray<double>& z) const
{
	return xt::maximum(a * z, z);
}

xt::xarray<double> LeakyReLUFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return leakyReLU(inputs);
}

xt::xarray<double> LeakyReLUFunction::getGradient(const xt::xarray<double>& sigmas) const
{
	auto mask = (lastOutput > 0.0);
	return (sigmas * (mask + (a * (xt::ones<double>(mask.shape()) - mask))));
}

double LeakyReLUFunction::getParameter(const std::string& parameterName) const
{
	if (parameterName.compare(LeakyReLUFunction::A) == 0)
	{
		return a;
	}
	else
	{
		throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
	}
}

void LeakyReLUFunction::setParameter(const std::string& parameterName, double value)
{
	if (parameterName.compare(LeakyReLUFunction::A) == 0)
	{
		a = value;
	}
	else
	{
		throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
	}
}

void LeakyReLUFunction::saveParameters(std::string fileName)
{
	xt::dump_npy(fileName + "_a.npy", xt::xarray<double>({ a }));
}

void LeakyReLUFunction::loadParameters(std::string fileName)
{
	bool exists = NeuralNetworkFileHelper::fileExists(fileName + "_a.npy");
	if (exists)
	{
		a = xt::load_npy<double>(fileName + "_a.npy")(0);
	}
	else
	{
		cout << "Parameter file " + fileName + "_a.npy" + "not found" << endl;
	}
}

double LeakyReLUFunction::getA() const
{ 
	return a;
}

void LeakyReLUFunction::setA(double a) 
{
	this->a = a;
}

void LeakyReLUFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
{
	ActivationFunction::draw(canvas, origin, scale, numUnits, weights);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	xt::xarray<double> drawWeights = weights.getParameters();

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = drawWeights(0, i);
		double inv_slope = (slope == 0) ? (0.0) : (1.0 / abs(slope));
		double x1, x2, y1, y2;
		if (slope > 0.0f)
		{
			x1 = -1.0;
			x2 = +min(1.0, inv_slope);
			y1 = -a;
			y2 = (x2 * slope);
		}
		else
		{
			x1 = -min(1.0, inv_slope);
			x2 = 1.0;
			y1 = (x1 * slope);
			y2 = a;
		}

		ImVec2 l_start(position.x + (NeuralLayer::DRAW_LEN * x1 * scale), position.y - (NeuralLayer::DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (NeuralLayer::DRAW_LEN * x2 * scale), position.y - (NeuralLayer::DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}