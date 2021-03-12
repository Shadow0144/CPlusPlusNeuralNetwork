#include "ParameterizedNeuralLayer.h"

#pragma warning(push, 0)
#include <xtensor/xnpy.hpp>
#pragma warning(pop)

using namespace std;

void ParameterizedNeuralLayer::saveParameters(std::string fileName)
{
	xt::dump_npy(fileName + ".npy", weights.getParameters());
}

void ParameterizedNeuralLayer::loadParameters(std::string fileName)
{
	bool exists = fileExists(fileName + ".npy");
	if (exists)
	{
		weights.setParameters(xt::load_npy<double>(fileName + ".npy"));
	}
	else
	{
		cout << "Parameter file " + fileName + ".npy" + "not found" << endl;
	}
}