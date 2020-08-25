#pragma once

#include "NeuralLayer.h"
#include "Function.h"
#include <vector>

using namespace std;

class InputNeuralLayer : public NeuralLayer
{
public:
	InputNeuralLayer(std::vector<size_t> inputShape);
	~InputNeuralLayer();

	xt::xarray<double> feedForward(xt::xarray<double> input);
	xt::xarray<double> backPropagate(xt::xarray<double> sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	NeuralLayer* children;
	std::vector<size_t> inputShape;

	xt::xarray<double> result; // Results of feedforward

	ImVec2 position; // For drawing

	void addChildren(NeuralLayer* children);

	friend class NetworkVisualizer;
};