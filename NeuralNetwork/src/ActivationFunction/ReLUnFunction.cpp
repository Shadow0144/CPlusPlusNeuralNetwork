#include "ActivationFunction/ReLUnFunction.h"
#include "NeuralLayer/NeuralLayer.h"
#include "NeuralNetworkFileHelper.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xnpy.hpp>
#pragma warning(pop)

using namespace std;

const std::string ReLUnFunction::N = "n"; // Parameter string [OPTIONAL]

ReLUnFunction::ReLUnFunction()
{

}

ReLUnFunction::ReLUnFunction(double n)
{
	this->n = n;
}

xt::xarray<double> ReLUnFunction::reLUn(const xt::xarray<double>& z) const
{
	return xt::minimum(xt::maximum(0.0, z), n);
}

xt::xarray<double> ReLUnFunction::feedForward(const xt::xarray<double>& inputs) const
{
	return reLUn(inputs);
}

xt::xarray<double> ReLUnFunction::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return (sigmas * ((lastOutput > 0.0) < n));
}

double ReLUnFunction::getParameter(const std::string& parameterName) const
{
	if (parameterName.compare(ReLUnFunction::N) == 0)
	{
		return n;
	}
	else
	{
		throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
	}
}

void ReLUnFunction::setParameter(const std::string& parameterName, double value)
{
	if (parameterName.compare(ReLUnFunction::N) == 0)
	{
		n = value;
	}
	else
	{
		throw std::invalid_argument(std::string("Parameter ") + parameterName + " does not exist");
	}
}

void ReLUnFunction::saveParameters(std::string fileName)
{
	xt::dump_npy(fileName + "_n.npy", xt::xarray<double>({ n }));
}

void ReLUnFunction::loadParameters(std::string fileName)
{
	bool exists = NeuralNetworkFileHelper::fileExists(fileName + "_n.npy");
	if (exists)
	{
		n = xt::load_npy<double>(fileName + "_n.npy")(0);
	}
	else
	{
		cout << "Parameter file " + fileName + "_n.npy" + "not found" << endl;
	}
}

double ReLUnFunction::getN() const
{
	return n;
}

void ReLUnFunction::setN(double n)
{
	this->n = n;
}

void ReLUnFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale, int numUnits, const ParameterSet& weights) const
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
		if (slope > 0.0)
		{
			x1 = -1.0;
			x2 = +min(1.0, inv_slope);
			y1 = 0.0;
			y2 = min((x2 * slope), n);
		}
		else
		{
			x1 = -min(1.0, inv_slope);
			x2 = 1.0;
			y1 = min((x1 * slope), n);
			y2 = 0.0;
		}

		ImVec2 l_start(position.x + (NeuralLayer::DRAW_LEN * x1 * scale), position.y - (NeuralLayer::DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (NeuralLayer::DRAW_LEN * x2 * scale), position.y - (NeuralLayer::DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}