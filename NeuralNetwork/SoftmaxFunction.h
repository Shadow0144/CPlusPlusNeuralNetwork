#pragma once

#include "Function.h"
#include <shared_mutex>

// Softmax
class SoftmaxFunction : public Function
{
public:
	SoftmaxFunction(size_t incomingUnits, int axis = -1);

	xt::xarray<double> feedForward(const xt::xarray<double>& input);
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& inputs); // Overriding to get a mutex lock
	xt::xarray<double> backPropagate(const xt::xarray<double>& sigmas);
	// Special case where the error function is cross entropy
	xt::xarray<double> backPropagateCrossEntropy(const xt::xarray<double>& sigmas);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	int axis;
	size_t numOutputs;

	mutable std::shared_mutex outputMutex;
};