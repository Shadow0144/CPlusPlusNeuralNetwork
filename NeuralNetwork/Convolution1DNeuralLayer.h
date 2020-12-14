#pragma once

#include "NeuralLayer.h"
#include "Function.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#pragma warning(pop)

class Convolution1DNeuralLayer : public NeuralLayer
{
public:
	Convolution1DNeuralLayer(NeuralLayer* parent, size_t numKernels,
		const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride = 1);
	~Convolution1DNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& errors);
	double applyBackPropagate();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	Function* activationFunction;

	size_t numKernels;
	std::vector<size_t> convolutionShape;
	size_t stride;
	size_t inputChannels;

	xt::xstrided_slice_vector kernelWindowView;

	ParameterSet weights;

	xt::xarray<double> convolude1D(const xt::xarray<double>& f, const xt::xarray<double>& g);
	void draw1DConvolution(ImDrawList* canvas, ImVec2 origin, double scale);
};