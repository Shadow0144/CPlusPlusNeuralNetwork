#pragma once

#include "ConvolutionNeuralLayer.h"
#include "ActivationFunction.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#include <map>
#pragma warning(pop)

class Convolution1DNeuralLayer : public ConvolutionNeuralLayer
{
public:
	Convolution1DNeuralLayer(NeuralLayer* parent, size_t numKernels,
		const std::vector<size_t>& convolutionShape, 
		size_t inputChannels, size_t stride = 1, bool addBias = false,
		ActivationFunctionType activationFunctionType = ActivationFunctionType::Identity,
		std::map<std::string, double> additionalParameters = std::map<std::string, double>());
	~Convolution1DNeuralLayer();

	xt::xarray<double> backPropagate(const xt::xarray<double>& errors);

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	xt::xarray<double> convolveInput(const xt::xarray<double>& input);
	xt::xarray<double> convolude1D(const xt::xarray<double>& f, const xt::xarray<double>& g);
	void draw1DConvolution(ImDrawList* canvas, ImVec2 origin, double scale);
};