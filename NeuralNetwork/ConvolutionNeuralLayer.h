#pragma once

#include "ParameterizedNeuralLayer.h"

class ConvolutionNeuralLayer : public ParameterizedNeuralLayer
{
public:
	virtual ~ConvolutionNeuralLayer() = 0; // This class is not intended to be directly instantiated

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	double applyBackPropagate();

	void saveParameters(std::string fileName);
	void loadParameters(std::string fileName);

protected:
	bool hasBias;
	ParameterSet biasWeights;

	size_t numKernels;
	std::vector<size_t> convolutionShape;
	size_t stride;
	size_t inputChannels;

	xt::xstrided_slice_vector kernelWindowView;

	virtual xt::xarray<double> convolveInput(const xt::xarray<double>& input) = 0;
};