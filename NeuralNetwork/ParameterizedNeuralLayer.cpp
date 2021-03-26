#include "ParameterizedNeuralLayer.h"
#include "NeuralNetworkFileHelper.h"
#include "Optimizer.h"

#pragma warning(push, 0)
#include <xtensor/xnpy.hpp>
#pragma warning(pop)

using namespace std;

void ParameterizedNeuralLayer::saveParameters(std::string fileName)
{
	xt::dump_npy(fileName + ".npy", weights.getParameters());
	NeuralLayer::saveParameters(fileName); // Handles the activation function
}

void ParameterizedNeuralLayer::loadParameters(std::string fileName)
{
	bool exists = NeuralNetworkFileHelper::fileExists(fileName + ".npy");
	if (exists)
	{
		weights.setParameters(xt::load_npy<double>(fileName + ".npy"));
	}
	else
	{
		cout << "Parameter file " + fileName + ".npy" + "not found" << endl;
	}
	NeuralLayer::loadParameters(fileName); // Handles the activation function
}

void ParameterizedNeuralLayer::substituteParameters(Optimizer* optimizer)
{
	optimizer->substituteParameters(weights);
	NeuralLayer::substituteParameters(optimizer);
}

void ParameterizedNeuralLayer::restoreParameters(Optimizer* optimizer)
{
	optimizer->restoreParameters(weights);
	NeuralLayer::restoreParameters(optimizer);
}