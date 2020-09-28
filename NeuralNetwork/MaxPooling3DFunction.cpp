#include "MaxPooling3DFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#pragma warning(pop)

using namespace std;

// TODO: Padding and dimensions
MaxPooling3DFunction::MaxPooling3DFunction(size_t filterSize, size_t stride)
{
	this->hasBias = false;
	this->filterSize = filterSize;
	this->stride = stride;
}

xt::xarray<double> MaxPooling3DFunction::feedForward(xt::xarray<double> inputs)
{
	lastInput = inputs;

	const int DIM1 = inputs.dimension() - 3;
	const int DIM2 = inputs.dimension() - 2;
	const int DIM3 = inputs.dimension() - 1;
	auto shape = inputs.shape();
	shape[DIM1] = ceil((shape[DIM1] - (filterSize - 1)) / stride);
	shape[DIM2] = ceil((shape[DIM2] - (filterSize - 1)) / stride);
	shape[DIM3] = ceil((shape[DIM3] - (filterSize - 1)) / stride);
	lastOutput = xt::xarray<double>(shape);

	xt::xstrided_slice_vector inputWindowView;
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f <= DIM3; f++)
	{
		inputWindowView.push_back(xt::all());
		outputWindowView.push_back(xt::all());
	}

	int l = 0;
	int m = 0;
	int n = 0;
	const int I = (inputs.shape()[DIM1] - filterSize + 1);
	const int J = (inputs.shape()[DIM2] - filterSize + 1);
	const int K = (inputs.shape()[DIM3] - filterSize + 1);
	for (int i = 0; i < I; i += stride)
	{
		inputWindowView[DIM1] = xt::range(i, i + filterSize);
		outputWindowView[DIM1] = l++; // Increment after assignment
		for (int j = 0; j < J; j += stride)
		{
			inputWindowView[DIM2] = xt::range(j, j + filterSize);
			outputWindowView[DIM2] = m++; // Increment after assignment
			for (int k = 0; k < K; k += stride)
			{
				inputWindowView[DIM3] = xt::range(k, k + filterSize);
				outputWindowView[DIM3] = n++; // Increment after assignment
				auto window = xt::xarray<double>(xt::strided_view(inputs, inputWindowView));
				xt::strided_view(lastOutput, outputWindowView) = xt::amax(window, { DIM1, DIM2, DIM3 });
			}
			n = 0;
		}
		m = 0;
	}

	return lastOutput;
}

xt::xarray<double> MaxPooling3DFunction::backPropagate(xt::xarray<double> sigmas)
{
	xt::xarray<double> sigmasPrime = xt::where(xt::equal(lastInput, lastOutput), 1, 0) * sigmas;
	return sigmasPrime;
}

void MaxPooling3DFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		double slope = weights.getParameters()(0, i);
		double inv_slope = 1.0 / abs(slope);
		double x1, x2, y1, y2;
		if (slope > 0.0)
		{
			x1 = -1.0;
			x2 = +min(1.0, inv_slope);
			y1 = 0.0;
			y2 = (x2 * slope);
		}
		else
		{
			x1 = -min(1.0, inv_slope);
			x2 = 1.0;
			y1 = (x1 * slope);
			y2 = 0.0;
		}

		ImVec2 l_start(position.x + (DRAW_LEN * x1 * scale), position.y - (DRAW_LEN * y1 * scale));
		ImVec2 l_mid(position.x, position.y);
		ImVec2 l_end(position.x + (DRAW_LEN * x2 * scale), position.y - (DRAW_LEN * y2 * scale));

		canvas->AddLine(l_start, l_mid, BLACK);
		canvas->AddLine(l_mid, l_end, BLACK);
	}
}