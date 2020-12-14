#pragma once

#include "NeuralLayer.h"
#include "Function.h"

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

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	void drawFlattenFunction(ImDrawList* canvas, ImVec2 origin, double scale);
};