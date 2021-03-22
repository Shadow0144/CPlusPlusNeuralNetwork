#pragma once

#pragma warning(push, 0)
#include "NeuralLayer.h"
#include <vector>
#pragma warning(pop)

using namespace std;

// During training, randomly scales each input by 0 dropRate% of the time, with all other inputs scaled by 1/(1-dropRate)
// During inference (i.e. not training), behaves as a pass-through or identity layer
class DropoutNeuralLayer : public NeuralLayer
{
public:
	DropoutNeuralLayer(NeuralLayer* parent);
	DropoutNeuralLayer(NeuralLayer* parent, double dropRate);
	~DropoutNeuralLayer();

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	double applyBackPropagate();

	double getDropRate();
	void setDropRate(double dropRate);

	std::vector<size_t> getOutputShape();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

private:
	double dropRate = 0.5;

	xt::xarray<double> lastMask;
};