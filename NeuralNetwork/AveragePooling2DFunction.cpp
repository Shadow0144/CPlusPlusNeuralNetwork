#include "AveragePooling2DFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#pragma warning(pop)

using namespace std;

// TODO: Padding and dimensions
AveragePooling2DFunction::AveragePooling2DFunction(std::vector<size_t> filterShape)
{
	this->hasBias = false;
	this->filterShape = filterShape;
}

xt::xarray<double> AveragePooling2DFunction::feedForward(xt::xarray<double> inputs)
{
	lastInput = inputs;

	const int DIM1 = inputs.dimension() - 3; // First dimension
	const int DIM2 = inputs.dimension() - 2; // Second dimension
	const int DIMC = inputs.dimension() - 1; // Channels
	auto shape = inputs.shape();
	shape[DIM1] = ceil((shape[DIM1] - (filterShape[0] - 1)) / filterShape[0]);
	shape[DIM2] = ceil((shape[DIM2] - (filterShape[1] - 1)) / filterShape[1]);
	lastOutput = xt::xarray<double>(shape);

	xt::xstrided_slice_vector inputWindowView;
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f <= DIMC; f++)
	{
		inputWindowView.push_back(xt::all());
		outputWindowView.push_back(xt::all());
	}

	int k = 0;
	int l = 0;
	const int I = (inputs.shape()[DIM1] - filterShape[0] + 1);
	const int J = (inputs.shape()[DIM2] - filterShape[1] + 1);
	for (int i = 0; i < I; i += filterShape[0])
	{
		inputWindowView[DIM1] = xt::range(i, i + filterShape[0]);
		outputWindowView[DIM1] = k++; // Increment after assignment
		for (int j = 0; j < J; j += filterShape[1])
		{
			inputWindowView[DIM2] = xt::range(j, j + filterShape[1]);
			outputWindowView[DIM2] = l++; // Increment after assignment
			auto window = xt::xarray<double>(xt::strided_view(inputs, inputWindowView));
			xt::strided_view(lastOutput, outputWindowView) = xt::mean(window, { DIM1, DIM2 });
		}
		l = 0;
	}

	return lastOutput;
}

xt::xarray<double> AveragePooling2DFunction::backPropagate(xt::xarray<double> sigmas)
{
	xt::xarray<double> sigmasPrime = xt::where(xt::equal(lastInput, lastOutput), 1, 0) * sigmas;
	return sigmasPrime;
}

void AveragePooling2DFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
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