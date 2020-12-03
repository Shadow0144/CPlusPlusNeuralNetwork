#pragma once

#include "NeuralLayer.h"
#include "Function.h"
#include <vector>

using namespace std;

class SqueezeNeuralLayer : public NeuralLayer
{
public:
	SqueezeNeuralLayer(const std::vector<size_t>& squeezeDims = std::vector<size_t>());
	~SqueezeNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	NeuralLayer* children;
	std::vector<size_t> squeezeDims;

	xt::xarray<double> result; // Results of feedforward

	ImVec2 position; // For drawing

	void addChildren(NeuralLayer* children);

	friend class NetworkVisualizer;
};
