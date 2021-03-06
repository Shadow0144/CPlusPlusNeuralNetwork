#define _USE_MATH_DEFINES

#include "Convolution1DNeuralLayer.h"

#pragma warning(push, 0)
#include <xtensor/xview.hpp>
#include <math.h>
#include <tuple>
#pragma warning(pop)

using namespace std;

Convolution1DNeuralLayer::Convolution1DNeuralLayer(NeuralLayer* parent, size_t numKernels,
	const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride)
{
	this->parent = parent;
	this->children = NULL;
	if (parent != NULL)
	{
		parent->addChildren(this);
	}
	else { }
	this->numUnits = numKernels;
	this->convolutionShape = convolutionShape;
	this->stride = stride;
	this->inputChannels = inputChannels;
	this->numKernels = numKernels;

	//this->hasBias = false; // TODO!
	// Add activation function too

	std::vector<size_t> paramShape;
	// convolution x ... x filters x kernel -shaped
	kernelWindowView = xt::xstrided_slice_vector();

	// 1D
	paramShape.push_back(convolutionShape[0]);
	kernelWindowView.push_back(xt::all()); // First dimension

	kernelWindowView.push_back(xt::all()); // Channels / Filters
	kernelWindowView.push_back(0); // Current kernel
	paramShape.push_back(inputChannels);
	paramShape.push_back(numKernels);
	this->weights.setParametersRandom(paramShape);
}

Convolution1DNeuralLayer::~Convolution1DNeuralLayer()
{

}

xt::xarray<double> Convolution1DNeuralLayer::convolude1D(const xt::xarray<double>& f, const xt::xarray<double>& g)
{
	// Assume the last dimension is the channel dimension
	const int DIMS = f.dimension();
	const int DIM1 = DIMS - 2;
	const int DIMC = DIMS - 1;

	auto kernelShape = g.shape(); // Assume g is the smaller kernel
	const int kDIMS = g.dimension();
	const int kDIM1 = kDIMS - 2;
	const int kDIMC = kDIMS - 1;

	auto shape = f.shape();
	shape[DIM1] = ceil((shape[DIM1] - (kernelShape[kDIM1] - 1)) / stride);
	shape.pop_back(); // Remove the last element
	xt::xarray<double> h = xt::xarray<double>(shape);

	xt::xstrided_slice_vector inputWindowView;
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f < DIMC; f++) // The output has only one channel
	{
		inputWindowView.push_back(xt::all());
		outputWindowView.push_back(xt::all());
	}
	inputWindowView.push_back(xt::all());

	const int I = (f.shape()[DIM1] - kernelShape[kDIM1] + 1);
	int x = 0;
	for (int i = 0; i < I; i += stride)
	{
		inputWindowView[DIM1] = xt::range(i, i + kernelShape[kDIM1]);
		outputWindowView[DIM1] = x++; // Increment after assignment
		auto window = xt::xarray<double>(xt::strided_view(f, inputWindowView));
		xt::strided_view(h, outputWindowView) = xt::sum(window * g, { DIM1, DIMC });
	}

	return h;
}

xt::xarray<double> Convolution1DNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	// Assume the last dimension is the channel dimension
	const int DIMS = input.dimension();
	const int DIM1 = DIMS - 2;
	const int DIMC = DIMS - 1;
	const int kDIMK = 2; // For a 1-D convolution, the kernels are the 3rd index

	// The output will have the same number of dimensions as the input
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f < DIMC; f++)
	{
		outputWindowView.push_back(xt::all());
	}
	outputWindowView.push_back(0); // Current filter

	// Set up the tensor to hold the result
	auto shape = input.shape();
	shape[DIM1] = ceil((shape[DIM1] - (convolutionShape[0] - 1)) / stride);
	shape[DIMC] = numKernels; // Output is potentially higher dimension
	xt::xarray<double> output = xt::xarray<double>(shape);

	for (int k = 0; k < numKernels; k++)
	{
		kernelWindowView[kDIMK] = k;
		outputWindowView[DIMC] = k; // Output is potentially higher dimension
		auto filter = xt::xarray<double>(xt::strided_view(weights.getParameters(), kernelWindowView));
		xt::strided_view(output, outputWindowView) = convolude1D(input, filter);
	}

	//output = xt::where(lastOutput > 0.0, lastOutput, 0); // TODO

	return output;
}

xt::xarray<double> Convolution1DNeuralLayer::backPropagate(const xt::xarray<double>& sigma)
{
	//sigmas *= (lastOutput > 0.0); // TODO: Replace with correct derivative code

	// The change in weights corresponds to a convolution between the input and the sigmas
	// Assume the last dimension is the channel dimension
	const int DIMS = lastInput.dimension();
	const int DIMN = 0;
	const int DIM1 = DIMS - 2;
	const int DIMC = DIMS - 1;
	const int kDIMF = 1; // For a 1-D convolution, the filters are the 2nd index
	const int kDIMK = 2; // For a 1-D convolution, the kernels are the 3rd index

	// The output will have the same number of dimensions as the input
	xt::xstrided_slice_vector sigmasWindowView;
	for (int f = 0; f < DIMC; f++)
	{
		sigmasWindowView.push_back(xt::all());
	}
	sigmasWindowView.push_back(0); // Current channel

	// Set up the tensor to hold the result
	xt::xarray<double> delta = xt::xarray<double>(weights.getDeltaParameters().shape());

	for (int k = 0; k < numKernels; k++)
	{
		// Select the kth sigma and the kth kernel to update
		sigmasWindowView[DIMC] = k;
		kernelWindowView[kDIMK] = k;
		auto kernelSigma = xt::xarray<double>(xt::strided_view(sigma, sigmasWindowView));
		// Repeat the sigma for each of the filters / channels
		kernelSigma = xt::expand_dims(kernelSigma, DIMC); // The output dimensionality should match the input
		kernelSigma = xt::repeat(kernelSigma, inputChannels, DIMC);
		// Convolude the last input with the sigmas for this kernel
		xt::xarray<double> result = convolude1D(lastInput, kernelSigma);
		// Repeat for each of the filters
		result = xt::expand_dims(result, DIMC);
		result = xt::repeat(result, inputChannels, DIMC);
		// Sum up initial dimensions until the correct size (e.g. the example dimension)
		while (result.dimension() > 2)
		{
			result = xt::sum(result, 0);
		}
		xt::strided_view(delta, kernelWindowView) = result;
	}

	weights.incrementDeltaParameters(-ALPHA * delta);

	//cout << "Delta: " << xt::sum(delta)(0) << endl;

	// The updated sigmas are a padded convolution between the original sigmas and the rotated filters
	// Pad the sigmas and repeat for each of the input channels
	auto inputShape = lastInput.shape();
	auto outputShape = lastOutput.shape();
	auto shape = lastOutput.shape();
	const int I = (inputShape[DIM1] - outputShape[DIM1]); // Padding size
	shape[DIM1] = (inputShape[DIM1] + I);
	shape.push_back(inputChannels);
	xt::xarray<double> paddedSigmas = xt::zeros<double>(shape);

	// For looking at the incoming sigmas and nesting them into a padded structure
	xt::xstrided_slice_vector paddedWindowView;
	for (int f = 0; f < DIM1; f++)
	{
		paddedWindowView.push_back(xt::all());
	}
	paddedWindowView.push_back(xt::range((I / 2), outputShape[DIM1] + (I / 2))); // First dimension
	paddedWindowView.push_back(xt::all()); // Filters
	paddedWindowView.push_back(xt::all()); // Kernels

	auto repSigmas = xt::repeat(xt::expand_dims(sigma, sigma.dimension()), inputChannels, sigma.dimension());
	xt::strided_view(paddedSigmas, paddedWindowView) = repSigmas;
	// Restore the view
	paddedWindowView[DIM1] = xt::all();

	// Create the rotated filters
	auto filters = weights.getParameters();
	xt::xarray<double> rotFilters = xt::xarray<double>(filters.shape());
	for (int i = 0; i < convolutionShape[0]; i++)
	{
		xt::view(rotFilters, convolutionShape[0] - i - 1, xt::all()) = 
			xt::view(filters, i, xt::all());
	}

	// We want to look at every kernel at once, and move through the filters instead
	kernelWindowView[kDIMK] = xt::all();

	xt::xstrided_slice_vector sigmasPrimeWindowView;
	for (int f = 0; f < DIMC; f++)
	{
		sigmasPrimeWindowView.push_back(xt::all());
	}
	sigmasPrimeWindowView.push_back(0); // Current channel

	// Perform the convolution
	// We want to look at each filter of all the kernels
	// There is one filter per input channel, but any number of kernels
	// The number of kernels will equal the number of output channels
	xt::xarray<double> sigmasPrime = xt::xarray<double>(inputShape);
	for (int c = 0; c < inputChannels; c++)
	{
		paddedWindowView[DIMC + 1] = c;
		kernelWindowView[kDIMF] = c;
		sigmasPrimeWindowView[DIMC] = c;
		auto filteredSigmas = xt::xarray<double>(xt::strided_view(paddedSigmas, paddedWindowView));
		auto kernels = xt::xarray<double>(xt::strided_view(rotFilters, kernelWindowView));
		auto result = convolude1D(filteredSigmas, kernels);
		xt::strided_view(sigmasPrime, sigmasPrimeWindowView) = result;
	}

	// Restore kernelWindowView
	kernelWindowView[kDIMF] = xt::all();

	return sigmasPrime;
}

double Convolution1DNeuralLayer::applyBackPropagate()
{
	double deltaWeight = xt::sum(xt::abs(weights.getDeltaParameters()))();
	weights.applyDeltaParameters();
	return deltaWeight; // Return the sum of how much the parameters have changed
}

void Convolution1DNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor VERY_LIGHT_GRAY(0.8f, 0.8f, 0.8f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);
	const double LINE_LENGTH = 15;
	const double WEIGHT_RADIUS = 10;
	const double BIAS_OFFSET_X = 40;
	const double BIAS_OFFSET_Y = -52;
	const double BIAS_FONT_SIZE = 24;
	const double BIAS_WIDTH = 20;
	const double BIAS_HEIGHT = BIAS_FONT_SIZE;
	const double BIAS_TEXT_X = 4;
	const double BIAS_TEXT_Y = 20;

	// Draw the neurons
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);
	}

	// Draw the convolution function
	draw1DConvolution(canvas, origin, scale);

	// Draw the links to the previous neurons
	double previousX, previousY;
	int parentCount = parent->getNumUnits();
	const double PARENT_LAYER_WIDTH = NeuralLayer::getLayerWidth(parentCount, scale);
	ImVec2 currentNeuronPt(0, origin.y - (RADIUS * scale));
	previousY = origin.y - (DIAMETER * scale);

	// Set up bias parameters
	double biasX = NeuralLayer::getNeuronX(origin.x, PARENT_LAYER_WIDTH, parentCount, scale);
	double biasY = previousY - RADIUS * scale;
	ImVec2 biasPt(biasX + 0.5 * (BIAS_WIDTH * scale), biasY + (BIAS_HEIGHT * scale));

	// Draw each neuron
	for (int i = 0; i < numUnits; i++)
	{
		currentNeuronPt.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		for (int j = 0; j < parentCount; j++) // There should be at least one parent
		{
			previousX = NeuralLayer::getNeuronX(origin.x, PARENT_LAYER_WIDTH, j, scale);
			ImVec2 previousNeuronPt(previousX, previousY);

			// Decide line color and width
			canvas->AddLine(previousNeuronPt, currentNeuronPt, BLACK, 1.0f);
		}
	} // for (int i = 0; i < numUnits; i++)

	if (output)
	{
		for (int i = 0; i < numUnits; i++)
		{
			// Draw the output lines
			double x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
			ImVec2 outputPt(x, position.y + (RADIUS * scale));
			ImVec2 nextPt(x, outputPt.y + (LINE_LENGTH * scale));
			canvas->AddLine(outputPt, nextPt, GRAY);
		}
	}
	else { }

	// Overlaying black ring
	for (int i = 0; i < numUnits; i++)
	{
		position.x = origin.x - (LAYER_WIDTH * 0.5) + (((DIAMETER + NEURON_SPACING) * i) * scale);
		canvas->AddCircle(position, RADIUS * scale, BLACK, 32);
	}
}

void Convolution1DNeuralLayer::draw1DConvolution(ImDrawList* canvas, ImVec2 origin, double scale)
{
	drawFunctionBackground(canvas, origin, scale, false);

	const int X = convolutionShape.at(0);
	const int Y = convolutionShape.at(1);

	const double RESCALE = DRAW_LEN * scale;
	double yHeight = 2.0 * RESCALE / Y;
	double xWidth = 2.0 * RESCALE / X;

	auto drawWeights = this->weights.getParameters();

	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	ImVec2 position(0, origin.y);
	for (int n = 0; n < numUnits; n++) // TODO: Fix padding issues
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, n, scale);
		int y = RESCALE - yHeight;
		for (int i = 0; i < Y; i++)
		{
			int x = -RESCALE;
			for (int j = 0; j < X; j++)
			{
				float colorValue = ((float)(drawWeights(i, j, 0, n)));
				ImColor color(colorValue, colorValue, colorValue);
				canvas->AddRectFilled(ImVec2(position.x + x, position.y - y),
					ImVec2(position.x + x + xWidth, position.y - y - yHeight),
					color);
				x += xWidth;
			}
			y -= yHeight;
		}
	}
}