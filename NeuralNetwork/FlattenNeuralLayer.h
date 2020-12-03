#pragma once

#include "NeuralLayer.h"
#include "Function.h"
#include "FlattenFunction.h"

#include <vector>

using namespace std;

class FlattenNeuralLayer : public NeuralLayer
{
public:
	FlattenNeuralLayer(NeuralLayer* parent, int numOutputs);
	~FlattenNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	NeuralLayer* parent;
	NeuralLayer* children;

	FlattenFunction* flattenFunction;

	ImVec2 position; // For drawing

	void addChildren(NeuralLayer* children);

	friend class NetworkVisualizer;
};