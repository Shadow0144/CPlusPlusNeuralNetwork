#pragma once

#include "NeuralLayer.h"
#include "Function.h"
#include "imgui.h"
#include <vector>

using namespace std;

class Convolution3DNeuralLayer : public NeuralLayer
{
public:
	Convolution3DNeuralLayer(NeuralLayer* parent, size_t numKernels, 
		const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride = 1);
	~Convolution3DNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& errors);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	Function* activationFunction;
	NeuralLayer* parent;
	NeuralLayer* children;
	std::vector<size_t> inputShape;
	xt::xarray<double> lastInput;
	xt::xarray<double> lastOutput;

	size_t numKernels;
	std::vector<size_t> convolutionShape;
	size_t stride;
	size_t inputChannels;

	xt::xstrided_slice_vector kernelWindowView;

	ParameterSet weights;
	const double ALPHA = 0.001; // Learning rate

	void addChildren(NeuralLayer* children);

	xt::xarray<double> convolude3D(const xt::xarray<double>& f, const xt::xarray<double>& g);
	void draw3DConvolution(ImDrawList* canvas, ImVec2 origin, double scale);
};