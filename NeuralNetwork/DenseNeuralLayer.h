#pragma once

#include "NeuralLayer.h"
#include "Function.h"
#include "imgui.h"
#include <vector>

using namespace std;

class DenseNeuralLayer : public NeuralLayer
{
public:
	DenseNeuralLayer(ActivationFunction function, NeuralLayer* parent, size_t numUnits);
	~DenseNeuralLayer();

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> errors);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	ActivationFunction functionType;
	Function* activationFunction;
	NeuralLayer* parent;
	NeuralLayer* children;
	std::vector<size_t> inputShape;

	xt::xarray<double> result; // Results of feedforward

	ImVec2 position; // For drawing
	
	void addChildren(NeuralLayer* children);

	friend class NetworkVisualizer;
};