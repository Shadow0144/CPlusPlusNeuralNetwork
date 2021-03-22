#pragma once

#pragma warning(push, 0)
#include "NeuralLayer.h"
#include <vector>
#pragma warning(pop)

using namespace std;

class InputNeuralLayer : public NeuralLayer
{
public:
	InputNeuralLayer(const std::vector<size_t>& inputShape);
	~InputNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);
};