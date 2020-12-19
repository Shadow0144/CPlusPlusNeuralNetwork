#pragma once

#include "NeuralLayer.h"
#include "ActivationFunction.h"

#pragma warning(push, 0)
#include <vector>
#pragma warning(pop)

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
	std::vector<size_t> squeezeDims;
};
