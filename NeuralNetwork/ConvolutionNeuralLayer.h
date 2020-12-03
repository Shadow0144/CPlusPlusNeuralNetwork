#pragma once

#include "NeuralLayer.h"
#include "Function.h"
#include "imgui.h"
#include <vector>

using namespace std;

class ConvolutionNeuralLayer : public NeuralLayer
{
public:
	ConvolutionNeuralLayer(ConvolutionActivationFunction function, NeuralLayer* parent, 
						size_t numKernels, const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride = 1);
	~ConvolutionNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& errors);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	ConvolutionActivationFunction functionType;
	Function* activationFunction;
	NeuralLayer* parent;
	NeuralLayer* children;
	std::vector<size_t> inputShape;

	void addChildren(NeuralLayer* children);
};