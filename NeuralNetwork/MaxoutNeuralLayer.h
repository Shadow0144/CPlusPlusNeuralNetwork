#pragma once

#include "NeuralLayer.h"
#include "ParameterSet.h"

#pragma warning(push, 0)
#include "imgui.h"
#include <vector>
#include <shared_mutex>
#pragma warning(pop)

class MaxoutNeuralLayer : public NeuralLayer
{
public:
	MaxoutNeuralLayer(NeuralLayer* parent, size_t numUnits, size_t numFunctions, bool addBias = true);
	~MaxoutNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	bool addBias;
	int numInputs;
	size_t numFunctions;

	ParameterSet weights;

	xt::xarray<double> maxMask; // For use by backpropagation

	xt::xarray<double> dotProduct(const xt::xarray<double>& input);
	xt::xarray<double> maxout(const xt::xarray<double>& input, bool storeIndices = false);

	void drawMaxout(ImDrawList* canvas, ImVec2 origin, double scale);
};