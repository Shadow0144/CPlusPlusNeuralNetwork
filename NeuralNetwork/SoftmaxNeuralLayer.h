#pragma once

#include "NeuralLayer.h"
#include "Function.h"
#include "imgui.h"
#include <vector>

using namespace std;

class SoftmaxNeuralLayer : public NeuralLayer
{
public:
	SoftmaxNeuralLayer(NeuralLayer* parent, int axis = -1);
	~SoftmaxNeuralLayer();

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> errors);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	Function* softmaxFunction;
	NeuralLayer* parent;
	NeuralLayer* children;
	std::vector<size_t> inputShape;
	size_t numOutputs;

	ParameterSet weights;

	bool hasBias;
	std::vector<int> sumIndices;
	xt::xarray<double> lastOutput;

	void addChildren(NeuralLayer* children);
};