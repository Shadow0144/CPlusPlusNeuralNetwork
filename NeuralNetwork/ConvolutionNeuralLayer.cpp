#include "ConvolutionNeuralLayer.h"
#include "NeuralNetworkFileHelper.h"

#pragma warning(push, 0)
#include <xtensor/xnpy.hpp>
#pragma warning(pop)

using namespace std;

ConvolutionNeuralLayer::~ConvolutionNeuralLayer()
{

}

xt::xarray<double> ConvolutionNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	return activationFunction->feedForward(convolveInput(input));
}

xt::xarray<double> ConvolutionNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	lastInput = input;
	lastOutput = activationFunction->feedForwardTrain(convolveInput(input));
	return lastOutput;
}

double ConvolutionNeuralLayer::applyBackPropagate()
{
	double deltaWeight = xt::sum(xt::abs(weights.getDeltaParameters()))();
	weights.applyDeltaParameters();
	if (hasBias)
	{
		deltaWeight += xt::sum(xt::abs(biasWeights.getDeltaParameters()))();
		biasWeights.applyDeltaParameters();
	}
	else { }
	deltaWeight += activationFunction->applyBackPropagate();
	return deltaWeight; // Return the sum of how much the parameters have changed
}

void ConvolutionNeuralLayer::saveParameters(std::string fileName)
{
	ParameterizedNeuralLayer::saveParameters(fileName); // Handles the wieghts and activation function
	if (hasBias)
	{
		xt::dump_npy(fileName + "_b.npy", biasWeights.getParameters());
	}
	else { }
}

void ConvolutionNeuralLayer::loadParameters(std::string fileName)
{
	ParameterizedNeuralLayer::loadParameters(fileName); // Handles the weights and activation function
	if (hasBias)
	{
		bool exists = NeuralNetworkFileHelper::fileExists(fileName + "_b.npy");
		if (exists)
		{
			weights.setParameters(xt::load_npy<double>(fileName + "_b.npy"));
		}
		else
		{
			cout << "Parameter file " + fileName + "_b.npy" + " not found" << endl;
		}
	}
	else { }
}