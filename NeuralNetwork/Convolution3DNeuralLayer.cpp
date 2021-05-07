#define _USE_MATH_DEFINES

#include "Convolution3DNeuralLayer.h"

#include "ActivationFunctionFactory.h"
#include "NetworkExceptions.h"

#pragma warning(push, 0)
#include <xtensor/xview.hpp>
#include <tuple>
#pragma warning(pop)

using namespace std;

Convolution3DNeuralLayer::Convolution3DNeuralLayer(NeuralLayer* parent, size_t numKernels,
	const std::vector<size_t>& convolutionShape, const std::vector<size_t>& stride, bool addBias,
	ActivationFunctionType activationFunctionType, std::map<string, double> additionalParameters)
	: ConvolutionNeuralLayer(parent, 3, numKernels, convolutionShape, stride, addBias,
		activationFunctionType, additionalParameters)
{

}

Convolution3DNeuralLayer::~Convolution3DNeuralLayer()
{

}

xt::xarray<double> Convolution3DNeuralLayer::convolude3D(const xt::xarray<double>& f, const xt::xarray<double>& g, bool useStride)
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

	int Is = (useStride) ? stride[0] : 1;
	int Js = (useStride) ? stride[1] : 1;
	int Ks = (useStride) ? stride[2] : 1;

	auto shape = f.shape();
	shape[DIM1] = ceil((shape[DIM1] - (kernelShape[kDIM1] - 1)) / Is);
	shape[DIM2] = ceil((shape[DIM2] - (kernelShape[kDIM2] - 1)) / Js);
	shape[DIM3] = ceil((shape[DIM3] - (kernelShape[kDIM3] - 1)) / Ks);
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
	for (int i = 0; i < I; i += Is)
	{
		int y = 0;
		inputWindowView[DIM1] = xt::range(i, i + kernelShape[kDIM1]);
		outputWindowView[DIM1] = x++; // Increment after assignment
		for (int j = 0; j < J; j += Js)
		{
			int z = 0;
			inputWindowView[DIM2] = xt::range(j, j + kernelShape[kDIM2]);
			outputWindowView[DIM2] = y++; // Increment after assignment
			for (int k = 0; k < K; k += Ks)
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
	shape[DIM1] = ceil((shape[DIM1] - (convolutionShape[0] - 1)) / stride[0]);
	shape[DIM2] = ceil((shape[DIM2] - (convolutionShape[1] - 1)) / stride[1]);
	shape[DIM3] = ceil((shape[DIM3] - (convolutionShape[2] - 1)) / stride[2]);
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

xt::xarray<double> Convolution3DNeuralLayer::getGradient(const xt::xarray<double>& sigma, Optimizer* optimizer)
{
	xt::xarray<double> actSigmas = activationFunction->getGradient(sigma, optimizer); // Pass the sigmas through the activation function first

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

	// In the case of stride > 1, the sigmas need to be dilated
	xt::xarray<double> dSigmas; // Dilated sigmas
	auto iShape = lastInput.shape();
	auto fShape = weights.getDeltaParameters().shape();
	auto dShape = actSigmas.shape();
	if (stride[0] > 1 || stride[1] > 1 || stride[2] > 1)
	{
		dShape[DIM1] = dShape[DIM1] + ((dShape[DIM1] - 1) * (stride[0] - 1)) + abs((int)(((iShape[DIM1] - fShape[0]) % stride[0])));
		dShape[DIM2] = dShape[DIM2] + ((dShape[DIM2] - 1) * (stride[1] - 1)) + abs((int)(((iShape[DIM2] - fShape[1]) % stride[1])));
		dShape[DIM3] = dShape[DIM3] + ((dShape[DIM3] - 1) * (stride[2] - 1)) + abs((int)(((iShape[DIM3] - fShape[2]) % stride[2])));
		dSigmas = xt::zeros<double>(dShape);

		int l = 0;
		for (int i = 0; i < dShape[DIM1]; i += stride[0])
		{
			int m = 0;
			for (int j = 0; j < dShape[DIM2]; j += stride[1])
			{
				int n = 0;
				for (int k = 0; k < dShape[DIM3]; k += stride[2])
				{
					xt::strided_view(dSigmas, { xt::ellipsis(), i, j, k, xt::all() }) = 
						xt::strided_view(actSigmas, { xt::ellipsis(), l, m, n, xt::all() });
					n++;
				}
				m++;
			}
			l++;
		}
	}
	else
	{
		dSigmas = actSigmas; // If stride is 1 in all dimensions, no dilation necessary
	}

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
		xt::xarray<double> result = convolude3D(lastInput, kernelSigma, false);
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

	optimizer->setDeltaWeight(weights, delta);
	if (hasBias)
	{
		optimizer->setDeltaWeight(biasWeights, delta);
	}
	else { }

	// The updated sigmas are a padded convolution between the original sigmas and the rotated filters
	// Pad the sigmas and repeat for each of the input channels
	auto inputShape = lastInput.shape();
	auto shape = lastOutput.shape();
	const int I = (fShape[0] - 1); // Padding size
	shape[DIM1] = (dShape[DIM1] + (2 * I));
	const int J = (fShape[1] - 1); // Padding size
	shape[DIM2] = (dShape[DIM2] + (2 * J));
	const int K = (fShape[2] - 1); // Padding size
	shape[DIM3] = (dShape[DIM3] + (2 * K));
	shape.push_back(inputChannels);
	xt::xarray<double> paddedSigmas = xt::zeros<double>(shape);

	// For looking at the incoming sigmas and nesting them into a padded structure
	xt::xstrided_slice_vector paddedWindowView;
	for (int f = 0; f < DIM1; f++)
	{
		paddedWindowView.push_back(xt::all());
	}
	paddedWindowView.push_back(xt::range(I, dShape[DIM1] + I)); // First dimension
	paddedWindowView.push_back(xt::range(J, dShape[DIM2] + J)); // Second dimension
	paddedWindowView.push_back(xt::range(K, dShape[DIM3] + K)); // Third dimension
	paddedWindowView.push_back(xt::all()); // Filters
	paddedWindowView.push_back(xt::all()); // Kernels

	// Use the dilated sigmas (in case of stride)
	auto repSigmas = xt::repeat(xt::expand_dims(dSigmas, dSigmas.dimension()), inputChannels, dSigmas.dimension());
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
		auto result = convolude3D(filteredSigmas, kernels, false);
		xt::strided_view(sigmasPrime, sigmasPrimeWindowView) = result;
	}

	// Restore kernelWindowView
	kernelWindowView[kDIMF] = xt::all();

	return sigmasPrime;
}

// TODO
void Convolution3DNeuralLayer::drawConvolution(ImDrawList* canvas, ImVec2 origin, double scale)
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