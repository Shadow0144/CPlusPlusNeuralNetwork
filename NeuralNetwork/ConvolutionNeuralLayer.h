#pragma once

#include "ParameterizedNeuralLayer.h"
#include "ActivationFunction.h"

class ConvolutionNeuralLayer : public ParameterizedNeuralLayer
{
public:
	void saveParameters(std::string fileName);
	void loadParameters(std::string fileName);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	double applyBackPropagate();

protected:
	ActivationFunction* activationFunction;
	bool hasBias;
	ParameterSet biasWeights;

	size_t numKernels;
	std::vector<size_t> convolutionShape;
	size_t stride;
	size_t inputChannels;

	xt::xstrided_slice_vector kernelWindowView;

	virtual xt::xarray<double> convolveInput(const xt::xarray<double>& input) = 0;
};