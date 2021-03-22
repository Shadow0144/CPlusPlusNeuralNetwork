#pragma once

#include "NeuralLayer.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#include <shared_mutex>
#pragma warning(pop)

class SoftmaxNeuralLayer : public NeuralLayer
{
public:
	SoftmaxNeuralLayer(NeuralLayer* parent, int axis = -1);
	~SoftmaxNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	// Special case where the error function is cross entropy
	xt::xarray<double> getGradientCrossEntropy(const xt::xarray<double>& sigmas, Optimizer* optimizer);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	size_t numInputs;
	size_t numOutputs;
	int axis;

	mutable std::shared_mutex outputMutex;

	std::vector<int> sumIndices;

	void drawSoftmax(ImDrawList* canvas, ImVec2 origin, double scale);
};