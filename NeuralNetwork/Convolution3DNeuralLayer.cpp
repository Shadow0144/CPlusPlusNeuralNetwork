#define _USE_MATH_DEFINES

#include "Convolution3DNeuralLayer.h"

#include "ActivationFunctionFactory.h"

#pragma warning(push, 0)
#include <xtensor/xview.hpp>
#include <math.h>
#include <tuple>
#pragma warning(pop)

using namespace std;

Convolution3DNeuralLayer::Convolution3DNeuralLayer(NeuralLayer* parent, size_t numKernels,
	const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride, bool addBias,
	ActivationFunctionType activationFunctionType, std::map<string, double> additionalParameters)
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

	this->hasBias = addBias;

	this->activationFunction = ActivationFunctionFactory::getNewActivationFunction(activationFunctionType, additionalParameters);

	std::vector<size_t> paramShape;
	// convolution x ... x filters x kernel -shaped
	kernelWindowView = xt::xstrided_slice_vector();

	// 3D
	paramShape.push_back(convolutionShape[0]);
	paramShape.push_back(convolutionShape[1]);
	paramShape.push_back(convolutionShape[2]);
	kernelWindowView.push_back(xt::all()); // First dimension
	kernelWindowView.push_back(xt::all()); // Second dimension
	kernelWindowView.push_back(xt::all()); // Third dimension

	kernelWindowView.push_back(xt::all()); // Channels / Filters
	kernelWindowView.push_back(0); // Current kernel
	paramShape.push_back(inputChannels);
	paramShape.push_back(numKernels);
	this->weights.setParametersRandom(paramShape);

	if (hasBias)
	{
		this->biasWeights.setParametersRandom(numKernels);
	}
	else { }
}

Convolution3DNeuralLayer::~Convolution3DNeuralLayer()
{

}

xt::xarray<double> Convolution3DNeuralLayer::convolude3D(const xt::xarray<double>& f, const xt::xarray<double>& g)
{
	// Assume the last dimension is the channel dimension
	const int DIMS = f.dimension();
	const int DIM1 = DIMS - 4;
	const int DIM2 = DIMS - 3;
	const int DIM3 = DIMS - 2;
	const int DIMC = DIMS - 1;

	auto kernelShape = g.shape(); // Assume g is the smaller kernel
	const int kDIMS = g.dimension();
	const int kDIM1 = kDIMS - 4;
	const int kDIM2 = kDIMS - 3;
	const int kDIM3 = kDIMS - 2;
	const int kDIMC = kDIMS - 1;

	auto shape = f.shape();
	shape[DIM1] = ceil((shape[DIM1] - (kernelShape[kDIM1] - 1)) / stride);
	shape[DIM2] = ceil((shape[DIM2] - (kernelShape[kDIM2] - 1)) / stride);
	shape[DIM3] = ceil((shape[DIM3] - (kernelShape[kDIM3] - 1)) / stride);
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
	const int J = (f.shape()[DIM2] - kernelShape[kDIM2] + 1);
	const int K = (f.shape()[DIM3] - kernelShape[kDIM3] + 1);
	int x = 0;
	for (int i = 0; i < I; i += stride)
	{
		int y = 0;
		inputWindowView[DIM1] = xt::range(i, i + kernelShape[kDIM1]);
		outputWindowView[DIM1] = x++; // Increment after assignment
		for (int j = 0; j < J; j += stride)
		{
			int z = 0;
			inputWindowView[DIM2] = xt::range(j, j + kernelShape[kDIM2]);
			outputWindowView[DIM2] = y++; // Increment after assignment
			for (int k = 0; k < K; k += stride)
			{
				inputWindowView[DIM3] = xt::range(k, k + kernelShape[kDIM3]);
				outputWindowView[DIM3] = z++; // Increment after assignment
				auto window = xt::xarray<double>(xt::strided_view(f, inputWindowView));
				xt::strided_view(h, outputWindowView) = xt::sum(window * g, { DIM1, DIM2, DIM3, DIMC });
			}
		}
	}

	return h;
}

xt::xarray<double> Convolution3DNeuralLayer::convolveInput(const xt::xarray<double>& input)
{
	// Assume the last dimension is the channel dimension
	const int DIMS = input.dimension();
	const int DIM1 = DIMS - 4;
	const int DIM2 = DIMS - 3;
	const int DIM3 = DIMS - 2;
	const int DIMC = DIMS - 1;
	const int kDIMK = 4; // For a 3-D convolution, the kernels are the 5th index

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
	shape[DIM2] = ceil((shape[DIM2] - (convolutionShape[1] - 1)) / stride);
	shape[DIM3] = ceil((shape[DIM3] - (convolutionShape[2] - 1)) / stride);
	shape[DIMC] = numKernels; // Output is potentially higher dimension
	xt::xarray<double> output = xt::xarray<double>(shape);

	for (int k = 0; k < numKernels; k++)
	{
		kernelWindowView[kDIMK] = k;
		outputWindowView[DIMC] = k; // Output is potentially higher dimension
		auto filter = xt::xarray<double>(xt::strided_view(weights.getParameters(), kernelWindowView));
		xt::strided_view(output, outputWindowView) = convolude3D(input, filter);
	}

	if (hasBias)
	{
		output += biasWeights.getParameters(); // Add bias
	}
	else { }

	return output;
}

xt::xarray<double> Convolution3DNeuralLayer::backPropagate(const xt::xarray<double>& sigma)
{
	xt::xarray<double> actSigma = activationFunction->getGradient(sigma); // Pass the sigmas through the activation function first

	// The change in weights corresponds to a convolution between the input and the sigmas
	// Assume the last dimension is the channel dimension
	const int DIMS = lastInput.dimension();
	const int DIMN = 0;
	const int DIM1 = DIMS - 4;
	const int DIM2 = DIMS - 3;
	const int DIM3 = DIMS - 2;
	const int DIMC = DIMS - 1;
	const int kDIMF = 3; // For a 3-D convolution, the filters are the 4th index
	const int kDIMK = 4; // For a 3-D convolution, the kernels are the 5th index

	// The output will have the same number of dimensions as the input
	xt::xstrided_slice_vector sigmasWindowView;
	for (int f = 0; f < DIMC; f++)
	{
		sigmasWindowView.push_back(xt::all());
	}
	sigmasWindowView.push_back(0); // Current channel

	// Set up the tensor to hold the result
	xt::xarray<double> delta = xt::xarray<double>(weights.getDeltaParameters().shape());
	xt::xarray<double> deltaBias;
	if (hasBias)
	{
		deltaBias = xt::xarray<double>(numKernels);
	}
	else { }

	for (int k = 0; k < numKernels; k++)
	{
		// Select the kth sigma and the kth kernel to update
		sigmasWindowView[DIMC] = k;
		kernelWindowView[kDIMK] = k;
		auto kernelSigma = xt::xarray<double>(xt::strided_view(actSigma, sigmasWindowView));
		if (hasBias) // The gradient for the bias is simply the sum of the sigmas for each kernel
		{
			deltaBias(k) = xt::sum(kernelSigma)();
		}
		else { }
		// Repeat the sigma for each of the filters / channels
		kernelSigma = xt::expand_dims(kernelSigma, DIMC); // The output dimensionality should match the input
		kernelSigma = xt::repeat(kernelSigma, inputChannels, DIMC);
		// Convolude the last input with the sigmas for this kernel
		xt::xarray<double> result = convolude3D(lastInput, kernelSigma);
		// Repeat for each of the filters
		result = xt::expand_dims(result, DIMC);
		result = xt::repeat(result, inputChannels, DIMC);
		// Sum up initial dimensions until the correct size (e.g. the example dimension)
		while (result.dimension() > 4)
		{
			result = xt::sum(result, 0);
		}
		xt::strided_view(delta, kernelWindowView) = result;
	}

	weights.incrementDeltaParameters(-ALPHA * delta);
	if (hasBias)
	{
		biasWeights.incrementDeltaParameters(-ALPHA * deltaBias);
	}
	else { }

	// The updated sigmas are a padded convolution between the original sigmas and the rotated filters
	// Pad the sigmas and repeat for each of the input channels
	auto inputShape = lastInput.shape();
	auto outputShape = lastOutput.shape();
	auto shape = lastOutput.shape();
	const int I = (inputShape[DIM1] - outputShape[DIM1]); // Padding size
	shape[DIM1] = (inputShape[DIM1] + I);
	const int J = (inputShape[DIM2] - shape[DIM2]);
	shape[DIM2] = (inputShape[DIM2] + J);
	const int K = (inputShape[DIM3] - shape[DIM3]);
	shape[DIM3] = (inputShape[DIM3] + K);
	shape.push_back(inputChannels);
	xt::xarray<double> paddedSigmas = xt::zeros<double>(shape);

	// For looking at the incoming sigmas and nesting them into a padded structure
	xt::xstrided_slice_vector paddedWindowView;
	for (int f = 0; f < DIM1; f++)
	{
		paddedWindowView.push_back(xt::all());
	}
	paddedWindowView.push_back(xt::range((I / 2), outputShape[DIM1] + (I / 2))); // First dimension
	paddedWindowView.push_back(xt::range((J / 2), outputShape[DIM2] + (J / 2))); // Second dimension
	paddedWindowView.push_back(xt::range((K / 2), outputShape[DIM3] + (K / 2))); // Third dimension
	paddedWindowView.push_back(xt::all()); // Filters
	paddedWindowView.push_back(xt::all()); // Kernels

	auto repSigmas = xt::repeat(xt::expand_dims(sigma, sigma.dimension()), inputChannels, sigma.dimension());
	xt::strided_view(paddedSigmas, paddedWindowView) = repSigmas;
	// Restore the view
	paddedWindowView[DIM1] = xt::all();
	paddedWindowView[DIM2] = xt::all();
	paddedWindowView[DIM3] = xt::all();

	// Create the rotated filters
	auto filters = weights.getParameters();
	xt::xarray<double> rotFilters = xt::xarray<double>(filters.shape());
	for (int i = 0; i < convolutionShape[0]; i++)
	{
		for (int j = 0; j < convolutionShape[1]; j++)
		{
			for (int k = 0; k < convolutionShape[2]; k++)
			{
				xt::view(rotFilters, convolutionShape[0] - i - 1, convolutionShape[1] - j - 1, convolutionShape[2] - k - 1, xt::all()) =
					xt::view(filters, i, j, k, xt::all());
			}
		}
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
		auto result = convolude3D(filteredSigmas, kernels);
		xt::strided_view(sigmasPrime, sigmasPrimeWindowView) = result;
	}

	// Restore kernelWindowView
	kernelWindowView[kDIMF] = xt::all();

	return sigmasPrime;
}

// TODO
void Convolution3DNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	// Draw the neurons
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);
	}

	// Draw the convolution function
	draw3DConvolution(canvas, origin, scale);

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

void Convolution3DNeuralLayer::draw3DConvolution(ImDrawList* canvas, ImVec2 origin, double scale)
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