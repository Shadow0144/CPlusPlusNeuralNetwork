#pragma once

#include "NeuralLayer/ParameterizedNeuralLayer.h"

class ConvolutionNeuralLayer : public ParameterizedNeuralLayer
{
public:
	virtual ~ConvolutionNeuralLayer() = 0; // This class is not intended to be directly instantiated

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	double applyBackPropagate();

	void saveParameters(std::string fileName);
	void loadParameters(std::string fileName);

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

protected:
	ConvolutionNeuralLayer(NeuralLayer* parent, size_t dims,
		size_t numKernels, const std::vector<size_t>& convolutionShape,
		const std::vector<size_t>& stride = { 1 }, const std::vector<size_t>& dilation = { 1 },
		bool padded = false, bool addBias = false,
		ActivationFunctionType activationFunctionType = ActivationFunctionType::Identity,
		std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	virtual void applyDilation() = 0;

	bool hasBias;
	ParameterSet biasWeights;

	size_t inputChannels;
	size_t numKernels;
	std::vector<size_t> convolutionShape;
	std::vector<size_t> stride;
	std::vector<size_t> dilation;
	bool padded;
	xt::svector<size_t> inputShape; // In case of padding

	xt::xstrided_slice_vector kernelWindowView;

	virtual xt::xarray<double> convolveInput(const xt::xarray<double>& input) = 0;

	xt::xarray<double> padInput(const xt::xarray<double>& input);
	xt::xarray<double> unpadSigmas(const xt::xarray<double>& sigmas);

	virtual void drawConvolution(ImDrawList* canvas, ImVec2 origin, double scale) = 0;
};