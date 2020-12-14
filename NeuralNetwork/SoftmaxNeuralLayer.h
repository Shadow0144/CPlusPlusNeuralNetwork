#pragma once

#include "NeuralLayer.h"
#include "imgui.h"
#include <vector>
#include <shared_mutex>

class SoftmaxNeuralLayer : public NeuralLayer
{
public:
	SoftmaxNeuralLayer(NeuralLayer* parent, int axis = -1);
	~SoftmaxNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	// Special case where the error function is cross entropy
	xt::xarray<double> backPropagateCrossEntropy(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	NeuralLayer* parent;
	NeuralLayer* children;
	std::vector<size_t> inputShape;
	size_t numInputs;
	size_t numOutputs;
	int axis;

	mutable std::shared_mutex outputMutex;

	std::vector<int> sumIndices;
	xt::xarray<double> lastInput;
	xt::xarray<double> lastOutput;

	void addChildren(NeuralLayer* children);
	void drawSoftmax(ImDrawList* canvas, ImVec2 origin, double scale);
};