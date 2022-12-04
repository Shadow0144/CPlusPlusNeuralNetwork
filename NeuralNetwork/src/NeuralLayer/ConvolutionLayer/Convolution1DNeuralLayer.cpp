#define _USE_MATH_DEFINES

#include "NeuralLayer/ConvolutionLayer/Convolution1DNeuralLayer.h"

#include "ActivationFunction/ActivationFunctionFactory.h"
#include "NetworkExceptions.h"

#pragma warning(push, 0)
#include <xtensor/xview.hpp>
#include <tuple>
#pragma warning(pop)

using namespace std;

Convolution1DNeuralLayer::Convolution1DNeuralLayer(NeuralLayer* parent, size_t numKernels,
	const std::vector<size_t>& convolutionShape, const std::vector<size_t>& stride, const std::vector<size_t>& dilation,
	bool padded, bool addBias, ActivationFunctionType activationFunctionType, std::map<string, double> additionalParameters)
	: ConvolutionNeuralLayer(parent, 1, numKernels, convolutionShape, stride, dilation, padded,
		addBias, activationFunctionType, additionalParameters)
{
	applyDilation();
}

Convolution1DNeuralLayer::~Convolution1DNeuralLayer()
{

}

void Convolution1DNeuralLayer::applyDilation()
{
	const int DIMS = 1;
	bool needsDilation = false;
	for (int i = 0; i < DIMS; i++)
	{
		if (dilation[i] > 1)
		{
			needsDilation = true;
			break;
		}
		else { }
	}
	if (needsDilation)
	{
		std::vector<size_t> paramShape;
		for (int i = 0; i < DIMS; i++)
		{
			paramShape.push_back(this->convolutionShape[i]);
		}
		paramShape.push_back(inputChannels);
		paramShape.push_back(numKernels);

		xt::xstrided_slice_vector dilatedWindowView;
		xt::xstrided_slice_vector undilatedWindowView;
		for (int f = 0; f < DIMS; f++)
		{
			dilatedWindowView.push_back(0);
			undilatedWindowView.push_back(0);
		}
		dilatedWindowView.push_back(xt::ellipsis());
		undilatedWindowView.push_back(xt::ellipsis());

		xt::xarray<double> dilatedWeights = xt::zeros<double>(paramShape);
		auto undilatedWeights = weights.getParameters();
		int l = 0;
		for (int i = 0; i < this->convolutionShape[0]; i += dilation[0])
		{
			dilatedWindowView[0] = i;
			undilatedWindowView[0] = l;
			xt::strided_view(dilatedWeights, dilatedWindowView) = xt::strided_view(undilatedWeights, undilatedWindowView);
			l++;
		}
		// l = 0;
		weights.setParameters(dilatedWeights);
	}
	else { }
}

xt::xarray<double> Convolution1DNeuralLayer::convolude1D(const xt::xarray<double>& f, const xt::xarray<double>& g, bool useStride)
{
	// Assume the last dimension is the channel dimension
	const int DIMS = f.dimension();
	const int DIM1 = DIMS - 2;
	const int DIMC = DIMS - 1;

	auto kernelShape = g.shape(); // Assume g is the smaller kernel
	const int kDIMS = g.dimension();
	const int kDIM1 = kDIMS - 2;
	const int kDIMC = kDIMS - 1;

	int Is = (useStride) ? stride[0] : 1;

	auto shape = f.shape();
	shape[DIM1] = ceil((shape[DIM1] - (kernelShape[kDIM1] - 1)) / ((double)(Is)));
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
	for (int i = 0; i < I; i += Is)
	{
		inputWindowView[DIM1] = xt::range(i, i + kernelShape[kDIM1]);
		outputWindowView[DIM1] = x++; // Increment after assignment
		auto window = xt::xarray<double>(xt::strided_view(f, inputWindowView));
		xt::strided_view(h, outputWindowView) = xt::sum(window * g, { DIM1, DIMC });
	}

	return h;
}

xt::xarray<double> Convolution1DNeuralLayer::convolveInput(const xt::xarray<double>& input)
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
	shape[DIM1] = ((shape[DIM1] - (convolutionShape[0] - 1)) / stride[0]);
	shape[DIMC] = numKernels; // Output is potentially higher dimension
	xt::xarray<double> output = xt::xarray<double>(shape);

	for (int k = 0; k < numKernels; k++)
	{
		kernelWindowView[kDIMK] = k;
		outputWindowView[DIMC] = k; // Output is potentially higher dimension
		auto filter = xt::xarray<double>(xt::strided_view(weights.getParameters(), kernelWindowView));
		xt::strided_view(output, outputWindowView) = convolude1D(input, filter);
	}

	if (hasBias)
	{
		output += biasWeights.getParameters(); // Add bias
	}
	else { }

	return output;
}

xt::xarray<double> Convolution1DNeuralLayer::getGradient(const xt::xarray<double>& sigma, Optimizer* optimizer)
{
	xt::xarray<double> actSigmas = activationFunction->getGradient(sigma, optimizer); // Pass the sigmas through the activation function first

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
	xt::xarray<double> deltaBias;
	if (hasBias)
	{
		deltaBias = xt::xarray<double>(numKernels);
	}
	else { }

	// In the case of stride > 1, the sigmas need to be dilated
	xt::xarray<double> dSigmas; // Dilated sigmas
	auto iShape = lastInput.shape();
	auto fShape = weights.getDeltaParameters().shape();
	auto dShape = actSigmas.shape();
	if (stride[0] > 1)
	{
		dShape[DIM1] = dShape[DIM1] + ((dShape[DIM1] - 1) * (stride[0] - 1)) + abs((int)(((iShape[DIM1] - fShape[0]) % stride[0])));
		dSigmas = xt::zeros<double>(dShape);

		int l = 0;
		for (int i = 0; i < dShape[DIM1]; i += stride[0])
		{
			xt::strided_view(dSigmas, { xt::ellipsis(), i, xt::all() }) =
				xt::strided_view(actSigmas, { xt::ellipsis(), l, xt::all() });
			l++;
		}
	}
	else
	{
		dSigmas = actSigmas; // If stride is 1 in all dimensions, no dilation necessary
	}

	// Weight update
	for (int k = 0; k < numKernels; k++)
	{
		// Select the kth sigma and the kth kernel to update
		sigmasWindowView[DIMC] = k;
		kernelWindowView[kDIMK] = k;
		auto kernelSigma = xt::xarray<double>(xt::strided_view(dSigmas, sigmasWindowView));
		if (hasBias) // The gradient for the bias is simply the sum of the sigmas for each kernel
		{
			deltaBias(k) = xt::sum(kernelSigma)();
		}
		else { }
		// Repeat the sigma for each of the filters / channels
		kernelSigma = xt::expand_dims(kernelSigma, DIMC); // The output dimensionality should match the input
		kernelSigma = xt::repeat(kernelSigma, inputChannels, DIMC);
		// Convolude the last input with the sigmas for this kernel
		xt::xarray<double> result = convolude1D(lastInput, kernelSigma, false);
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

	optimizer->setDeltaWeight(weights, delta);
	if (hasBias)
	{
		optimizer->setDeltaWeight(biasWeights, deltaBias);
	}
	else { }

	// The updated sigmas are a padded convolution between the original sigmas and the rotated filters
	// Pad the sigmas and repeat for each of the input channels
	auto inputShape = lastInput.shape();
	auto shape = lastOutput.shape();
	const int I = (fShape[0] - 1); // Padding size
	shape[DIM1] = (dShape[DIM1] + (2 * I));
	shape.push_back(inputChannels);
	xt::xarray<double> paddedSigmas = xt::zeros<double>(shape);

	// For looking at the incoming sigmas and nesting them into a padded structure
	xt::xstrided_slice_vector paddedWindowView;
	for (int f = 0; f < DIM1; f++)
	{
		paddedWindowView.push_back(xt::all());
	}
	paddedWindowView.push_back(xt::range(I, dShape[DIM1] + I)); // First dimension
	paddedWindowView.push_back(xt::all()); // Filters
	paddedWindowView.push_back(xt::all()); // Kernels

	// Use the dilated sigmas (in case of stride)
	auto repSigmas = xt::repeat(xt::expand_dims(dSigmas, dSigmas.dimension()), inputChannels, dSigmas.dimension());
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
		auto result = convolude1D(filteredSigmas, kernels, false);
		xt::strided_view(sigmasPrime, sigmasPrimeWindowView) = result;
	}

	// Restore kernelWindowView
	kernelWindowView[kDIMF] = xt::all();

	// Remove the padding if the input was padded
	if (padded)
	{
		sigmasPrime = unpadSigmas(sigmasPrime);
	}
	else { }

	return sigmasPrime;
}

void Convolution1DNeuralLayer::drawConvolution(ImDrawList* canvas, ImVec2 origin, double scale)
{
	drawFunctionBackground(canvas, origin, scale, false);

	const int X = convolutionShape.at(0);
	const int Y = 1;

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