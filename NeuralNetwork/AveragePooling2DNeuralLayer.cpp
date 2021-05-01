#include "AveragePooling2DNeuralLayer.h"

#include "NetworkExceptions.h"

#pragma warning(push, 0)
#include <math.h>
#include <tuple>
#pragma warning(pop)

using namespace std;

AveragePooling2DNeuralLayer::AveragePooling2DNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape, bool hasChannels)
	: PoolingNeuralLayer(parent, filterShape, hasChannels)
{
	if (filterShape.size() != 2)
	{
		throw NeuralLayerPoolingShapeException();
	}
	else { }
}

AveragePooling2DNeuralLayer::~AveragePooling2DNeuralLayer()
{

}

xt::xarray<double> AveragePooling2DNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	const int C = (hasChannels) ? 2 : 1;
	const int DIMS = input.dimension();
	const int DIM1 = DIMS - C - 1; // First dimension
	const int DIM2 = DIMS - C; // Second dimension
	const int DIMC = DIMS - 1; // Channels
	auto shape = input.shape();
	auto maxShape = xt::svector<size_t>(shape);
	shape[DIM1] = ceil(shape[DIM1] / filterShape[0]);
	shape[DIM2] = ceil(shape[DIM2] / filterShape[1]);
	xt::xarray<double> output = xt::xarray<double>(shape);
	maxShape[DIM1] = 1;
	maxShape[DIM2] = 1;

	xt::xstrided_slice_vector inputWindowView;
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f <= DIMC; f++)
	{
		inputWindowView.push_back(xt::all());
		outputWindowView.push_back(xt::all());
	}

	int l = 0;
	int m = 0;
	const int I = input.shape()[DIM1];
	const int J = input.shape()[DIM2];
	for (int i = 0; i < I; i += filterShape[0])
	{
		inputWindowView[DIM1] = xt::range(i, i + filterShape[0]);
		outputWindowView[DIM1] = l++; // Increment after assignment
		for (int j = 0; j < J; j += filterShape[1])
		{
			inputWindowView[DIM2] = xt::range(j, j + filterShape[1]);
			outputWindowView[DIM2] = m++; // Increment after assignment
			// Window contains subset of width and height and all channels of the input
			auto window = xt::xarray<double>(xt::strided_view(input, inputWindowView));
			// Reduce the w x h x c window to 1 x 1 x c
			auto maxes = xt::xarray<double>(xt::mean(window, { DIM1, DIM2 }));
			xt::strided_view(output, outputWindowView) = maxes;
		}
		m = 0;
	}
	// l = 0;

	return output;
}

xt::xarray<double> AveragePooling2DNeuralLayer::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	// Reverse what was done in feedforward, the input is now the output
	const int DIMS = lastInput.dimension();
	const int DIM1 = DIMS - 3; // First dimension
	const int DIM2 = DIMS - 2; // Second dimension
	const int DIMC = DIMS - 1; // Channels
	auto sigmaShape = sigmas.shape();
	sigmaShape[DIM1] = 1;
	sigmaShape[DIM2] = 1;

	xt::xarray<double> sigmasPrime = xt::zeros<double>(lastInput.shape());

	xt::xstrided_slice_vector primeWindowView;
	xt::xstrided_slice_vector sigmaWindowView;
	for (int f = 0; f <= DIMC; f++)
	{
		primeWindowView.push_back(xt::all());
		sigmaWindowView.push_back(xt::all());
	}

	int l = 0;
	int m = 0;
	const int I = lastInput.shape()[DIM1];
	const int J = lastInput.shape()[DIM2];
	for (int i = 0; i < I; i += filterShape[0])
	{
		primeWindowView[DIM1] = xt::range(i, i + filterShape[0]);
		sigmaWindowView[DIM1] = l++; // Increment after assignment
		for (int j = 0; j < J; j += filterShape[1])
		{
			primeWindowView[DIM2] = xt::range(j, j + filterShape[1]);
			sigmaWindowView[DIM2] = m++; // Increment after assignment
			auto sigma = xt::xarray<double>(xt::strided_view(sigmas, sigmaWindowView));
			sigma = xt::expand_dims(sigma, DIM1);
			sigma = xt::repeat(sigma, filterShape[0], DIM1);
			sigma = xt::expand_dims(sigma, DIM2);
			sigma = xt::repeat(sigma, filterShape[1], DIM2);
			auto sigmaExp = (((sigma) / I) / J);
			xt::strided_view(sigmasPrime, primeWindowView) += sigmaExp;
		}
		m = 0;
	}
	// l = 0;

	return sigmasPrime;
}

double AveragePooling2DNeuralLayer::applyBackPropagate()
{
	double deltaWeight = xt::sum(xt::abs(weights.getDeltaParameters()))();
	weights.applyDeltaParameters();
	return deltaWeight; // Return the sum of how much the parameters have changed
}

void AveragePooling2DNeuralLayer::drawPooling(ImDrawList* canvas, ImVec2 origin, double scale)
{
	drawConversionFunctionBackground(canvas, origin, scale, false);

	const int X = filterShape.at(0);
	const int Y = filterShape.at(1);

	const double RESCALE = DRAW_LEN * scale * RERESCALE;
	double yHeight = 2.0 * RESCALE / Y;
	double xWidth = 2.0 * RESCALE / X;

	const float CENTER_X = max(ceil((X - 1.0f) / 2.0f), 1.0f); // Avoid divide-by-zero
	const float CENTER_Y = max(ceil((Y - 1.0f) / 2.0f), 1.0f); // Avoid divide-by-zero

	float avgColor = 0.0f;
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
				avgColor += colorValue;
				ImColor color(colorValue, colorValue, colorValue);
				canvas->AddRectFilled(ImVec2(floor(position.x + x), floor(position.y - y)),
					ImVec2(ceil(position.x + x + xWidth), ceil(position.y - y - yHeight)),
					color);
				x += xWidth;
			}
			y -= yHeight;
		}
		avgColor /= (X * Y);

		// Draw right
		// Draw a grey square
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, n, scale) + (SHIFT * scale) - RESCALE;
		position.y = position.y + RESCALE;
		const ImColor HALF_GRAY(avgColor, avgColor, avgColor, 1.0f);
		canvas->AddRectFilled(ImVec2(floor(position.x), floor(position.y)),
			ImVec2(ceil(position.x + (xWidth * X)), ceil(position.y - (yHeight * Y))),
			HALF_GRAY);
	}
}