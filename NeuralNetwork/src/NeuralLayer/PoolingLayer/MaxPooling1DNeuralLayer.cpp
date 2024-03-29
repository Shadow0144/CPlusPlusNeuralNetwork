#include "NeuralLayer/PoolingLayer/MaxPooling1DNeuralLayer.h"

#include "NetworkExceptions.h"

#pragma warning(push, 0)
#include <math.h>
#include <tuple>
#pragma warning(pop)

using namespace std;

MaxPooling1DNeuralLayer::MaxPooling1DNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape,
													const std::vector<size_t>& stride, bool hasChannels)
	: PoolingNeuralLayer(parent, 1, filterShape, stride, hasChannels)
{

}

MaxPooling1DNeuralLayer::~MaxPooling1DNeuralLayer()
{

}

xt::xarray<double> MaxPooling1DNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	const int C = (hasChannels) ? 2 : 1;
	const int DIMS = input.dimension();
	const int DIM1 = DIMS - C; // First dimension
	const int DIMC = DIMS - 1; // Channels
	auto shape = input.shape();
	auto maxShape = xt::svector<size_t>(shape);
	shape[DIM1] = ceil((shape[DIM1] - (filterShape[0] - 1)) / ((double)(stride[0])));
	xt::xarray<double> output = xt::xarray<double>(shape);
	maxShape[DIM1] = 1;

	xt::xstrided_slice_vector inputWindowView;
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f <= DIMC; f++)
	{
		inputWindowView.push_back(xt::all());
		outputWindowView.push_back(xt::all());
	}

	int l = 0;
	auto iShape = input.shape();
	const int I = iShape[DIM1] - (iShape[DIM1] % filterShape[0]);
	for (int i = 0; i < I; i += filterShape[0])
	{
		inputWindowView[DIM1] = xt::range(i, i + filterShape[0]);
		outputWindowView[DIM1] = l++; // Increment after assignment
		// Window contains subset of width and height and all channels of the input
		auto window = xt::xarray<double>(xt::strided_view(input, inputWindowView));
		// Reduce the w x c window to 1 x c
		auto maxes = xt::xarray<double>(xt::amax(window, { DIM1 }));
		xt::strided_view(output, outputWindowView) = maxes;
	}
	// l = 0;

	return output;
}

xt::xarray<double> MaxPooling1DNeuralLayer::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	// Reverse what was done in feedforward, the input is now the output
	const int DIMS = lastInput.dimension();
	const int DIM1 = DIMS - 2; // First dimension
	const int DIMC = DIMS - 1; // Channels
	auto shape = lastInput.shape();
	auto maxesShape = lastInput.shape();
	shape[DIM1] = shape[DIM1] / filterShape[0];
	maxesShape[DIM1] = 1;
	auto sigmaShape = sigmas.shape();
	sigmaShape[DIM1] = 1;

	auto inputMask = xt::xarray<double>(shape); // Same shape as the last input

	xt::xarray<double> sigmasPrime = xt::zeros<double>(lastInput.shape());

	xt::xstrided_slice_vector primeWindowView;
	xt::xstrided_slice_vector sigmaWindowView;
	for (int f = 0; f <= DIMC; f++)
	{
		primeWindowView.push_back(xt::all());
		sigmaWindowView.push_back(xt::all());
	}

	int l = 0;
	auto iShape = lastInput.shape();
	const int I = iShape[DIM1] - (iShape[DIM1] % filterShape[0]);
	for (int i = 0; i < I; i += filterShape[0])
	{
		primeWindowView[DIM1] = xt::range(i, i + filterShape[0]);
		sigmaWindowView[DIM1] = l++; // Increment after assignment
		auto window = xt::strided_view(lastInput, primeWindowView);
		xt::xarray<double> maxes = xt::xarray<double>(xt::strided_view(lastOutput, sigmaWindowView));
		maxes = xt::expand_dims(maxes, DIM1);
		maxes = xt::repeat(maxes, filterShape[0], DIM1);
		auto sigma = xt::xarray<double>(xt::strided_view(sigmas, sigmaWindowView));
		sigma = xt::expand_dims(sigma, DIM1);
		sigma = xt::repeat(sigma, filterShape[0], DIM1);
		auto sigmaExp = xt::where(xt::equal(window, maxes), sigma, 0);
		xt::strided_view(sigmasPrime, primeWindowView) += sigmaExp;
	}
	// l = 0;

	return sigmasPrime;
}

double MaxPooling1DNeuralLayer::applyBackPropagate()
{
	double deltaWeight = xt::sum(xt::abs(weights.getDeltaParameters()))();
	weights.applyDeltaParameters();
	return deltaWeight; // Return the sum of how much the parameters have changed
}

void MaxPooling1DNeuralLayer::drawPooling(ImDrawList* canvas, ImVec2 origin, double scale)
{
	drawConversionFunctionBackground(canvas, origin, scale, false);

	const int X = filterShape.at(0);
	const int Y = 1; // Only one

	const double RESCALE = DRAW_LEN * scale * RERESCALE;
	double yHeight = 2.0 * RESCALE / Y;
	double xWidth = 2.0 * RESCALE / X;

	const float CENTER_X = max(ceil((X - 1.0f) / 2.0f), 1.0f); // Avoid divide-by-zero
	const float CENTER_Y = max(ceil((Y - 1.0f) / 2.0f), 1.0f); // Avoid divide-by-zero

	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	ImVec2 position(0, origin.y);
	for (int n = 0; n < numUnits; n++) // TODO: Fix padding issues
	{
		// Draw left
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, n, scale) - (SHIFT * scale);
		int y = RESCALE - yHeight;
		for (int i = 0; i < Y; i++)
		{
			int x = -RESCALE;
			for (int j = 0; j < X; j++)
			{
				float colorValue = 1.0f - (((abs(CENTER_X - j) / CENTER_X)
					+ (abs(CENTER_Y - i) / CENTER_Y)) / 2.0f);
				ImColor color(colorValue, colorValue, colorValue);
				canvas->AddRectFilled(ImVec2(floor(position.x + x), floor(position.y - y)),
					ImVec2(ceil(position.x + x + xWidth), ceil(position.y - y - yHeight)),
					color);
				x += xWidth;
			}
			y -= yHeight;
		}

		// Draw right
		// The blank grid is fine
	}
}