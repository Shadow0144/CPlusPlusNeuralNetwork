#pragma once

#include "NeuralLayer.h"
#include "Function.h"
#include "imgui.h"
#include <vector>

using namespace std;

class PoolingNeuralLayer : public NeuralLayer
{
public:
	PoolingNeuralLayer(PoolingActivationFunction function, NeuralLayer* parent, std::vector<size_t> filterShape);
	~PoolingNeuralLayer();

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> feedForwardTrain(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	PoolingActivationFunction functionType;
	Function* activationFunction;
	NeuralLayer* parent;
	NeuralLayer* children;
	std::vector<size_t> inputShape;

	void addChildren(NeuralLayer* children);
};
