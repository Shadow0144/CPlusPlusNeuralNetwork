#include "Convolution2DFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xmanipulation.hpp>
#pragma warning(pop)

#include "Test.h"

using namespace std;

Convolution2DFunction::Convolution2DFunction(std::vector<size_t> convolutionShape, size_t inputChannels, size_t stride, size_t numKernels)
{
	this->hasBias = false;
	this->numUnits = numKernels;
	this->convolutionShape = convolutionShape;
	this->stride = stride;
	this->inputChannels = inputChannels;
	this->numKernels = numKernels;
	this->drawAxes = false;

	std::vector<size_t> paramShape;
	// convolution x ... x filters x kernel -shaped
	for (int i = 0; i < convolutionShape.size(); i++)
	{
		paramShape.push_back(convolutionShape[i]);
	}
	paramShape.push_back(inputChannels);
	paramShape.push_back(numKernels);
	this->weights.setParametersRandom(paramShape);

	kernelWindowView = xt::xstrided_slice_vector();
	kernelWindowView.push_back(xt::all()); // First dimension
	kernelWindowView.push_back(xt::all()); // Second dimension
	kernelWindowView.push_back(xt::all()); // Channels / Filters
	kernelWindowView.push_back(0); // Current kernel
}

xt::xarray<double> Convolution2DFunction::convolude(xt::xarray<double> f, xt::xarray<double> g)
{
	// Assume the last dimension is the channel dimension
	const int DIMS = f.dimension();
	const int DIM1 = DIMS - 3;
	const int DIM2 = DIMS - 2;
	const int DIMC = DIMS - 1;

	auto kernelShape = g.shape(); // Assume g is the smaller kernel
	const int kDIMS = g.dimension();
	const int kDIM1 = kDIMS - 3;
	const int kDIM2 = kDIMS - 2;
	const int kDIMC = kDIMS - 1;

	auto shape = f.shape();
	shape[DIM1] = ceil((shape[DIM1] - (kernelShape[kDIM1] - 1)) / stride);
	shape[DIM2] = ceil((shape[DIM2] - (kernelShape[kDIM2] - 1)) / stride);
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
	int x = 0;
	for (int i = 0; i < I; i += stride)
	{
		inputWindowView[DIM1] = xt::range(i, i + kernelShape[kDIM1]);
		outputWindowView[DIM1] = x++; // Increment after assignment
		int y = 0;
		for (int j = 0; j < J; j += stride)
		{
			inputWindowView[DIM2] = xt::range(j, j + kernelShape[kDIM2]);
			outputWindowView[DIM2] = y++; // Increment after assignment
			auto window = xt::xarray<double>(xt::strided_view(f, inputWindowView));
			xt::strided_view(h, outputWindowView) = xt::sum(window * g);
		}
	}

	return h;
}

xt::xarray<double> Convolution2DFunction::feedForward(xt::xarray<double> inputs)
{
	// Assume the last dimension is the channel dimension
	const int DIMS = inputs.dimension();
	const int DIM1 = DIMS - 3;
	const int DIM2 = DIMS - 2;
	const int DIMC = DIMS - 1;
	const int kDIMK = 3; // For a 2-D convolution, the kernels are the 4th index

	// The output will have the same number of dimensions as the input
	xt::xstrided_slice_vector outputWindowView;
	for (int f = 0; f < DIMC; f++)
	{
		outputWindowView.push_back(xt::all());
	}
	outputWindowView.push_back(0); // Current filter

	// Set up the tensor to hold the result
	auto shape = inputs.shape();
	shape[DIM1] = ceil((shape[DIM1] - (convolutionShape[0] - 1)) / stride);
	shape[DIM2] = ceil((shape[DIM2] - (convolutionShape[1] - 1)) / stride);
	shape[DIMC] = numKernels; // Output is potentially higher dimension
	xt::xarray<double> output = xt::xarray<double>(shape);

	for (int k = 0; k < numKernels; k++)
	{
		kernelWindowView[kDIMK] = k; 
		outputWindowView[DIMC] = k; // Output is potentially higher dimension
		auto filter = xt::xarray<double>(xt::strided_view(weights.getParameters(), kernelWindowView));
		xt::strided_view(output, outputWindowView) = convolude(inputs, filter);
	}

	//output = xt::where(lastOutput > 0.0, lastOutput, 0); // TODO

	return output;
}

xt::xarray<double> Convolution2DFunction::backPropagate(xt::xarray<double> sigmas)
{
	//sigmas *= (lastOutput > 0.0); // TODO: Replace with correct derivative code

	// The change in weights corresponds to a convolution between the input and the sigmas

	// Assume the last dimension is the channel dimension
	const int DIMS = lastInput.dimension();
	const int DIM1 = DIMS - 3;
	const int DIM2 = DIMS - 2;
	const int DIMC = DIMS - 1;
	const int kDIMF = 2; // For a 2-D convolution, the filters are the 3rd index
	const int kDIMK = 3; // For a 2-D convolution, the kernels are the 4th index

	//print_dims(sigmas);
	//for (int i = 0; i < sigmas.shape()[0]; i++) // N
	//{
	//	for (int j = 0; j < sigmas.shape()[1]; j++) // x
	//	{
	//		for (int k = 0; k < sigmas.shape()[2]; k++) // y
	//		{
	//			for (int l = 0; l < sigmas.shape()[3]; l++) // c
	//			{
	//				cout << sigmas(i, j, k, l) << ", ";
	//			}
	//			cout << "] ";
	//		}
	//		cout << "} ";
	//	}
	//	cout << endl;
	//}

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
		auto filterSigma = xt::xarray<double>(xt::strided_view(sigmas, sigmasWindowView));
		// Repeat the sigma for each of the filters / channels
		filterSigma = xt::expand_dims(filterSigma, filterSigma.dimension());
		filterSigma = xt::repeat(filterSigma, inputChannels, filterSigma.dimension()-1);
		//print_dims(filterSigma);
		auto result = xt::repeat(xt::expand_dims(sum(convolude(lastInput, filterSigma), 0), 2), inputChannels, 2);
		/*print_dims(result);
		for (int i = 0; i < result.shape()[0]; i++)
		{
			for (int j = 0; j < result.shape()[1]; j++)
			{
				cout << result(i, j, 0, 0) << " ";
			}
			cout << endl;
		}*/
		xt::strided_view(delta, kernelWindowView) = result;
	}

	/*for (int i = 0; i < convolutionShape[0]; i++)
	{
		for (int j = 0; j < convolutionShape[1]; j++)
		{
			cout << delta(i, j, 0, 0) << " ";
		}
		cout << endl;
	}*/
	weights.incrementDeltaParameters(-ALPHA * delta);

	//cout << "Delta: " << xt::sum(delta)(0) << endl;

	// The updated sigmas are a padded convolution between the original sigmas and the rotated filters
	
	// Pad the sigmas and repeat for each of the input channels
	auto inputShape = lastInput.shape();
	auto outputShape = lastOutput.shape();
	auto shape = lastOutput.shape();
	const int I = (inputShape[DIM1] - outputShape[DIM1]); // Padding size
	shape[DIM1] = (inputShape[DIM1] + I);
	const int J = (inputShape[DIM2] - shape[DIM2]);
	shape[DIM2] = (inputShape[DIM2] + J);
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
	paddedWindowView.push_back(xt::all()); // Filters
	paddedWindowView.push_back(xt::all()); // Kernels
	
	auto repSigmas = xt::repeat(xt::expand_dims(sigmas, sigmas.dimension()), inputChannels, sigmas.dimension());
	xt::strided_view(paddedSigmas, paddedWindowView) = repSigmas;
	// Restore the view
	paddedWindowView[DIM1] = xt::all();
	paddedWindowView[DIM2] = xt::all();

	// Create the rotated filters
	auto filters = weights.getParameters();
	xt::xarray<double> rotFilters = xt::xarray<double>(filters.shape()); 
	for (int i = 0; i < convolutionShape[0]; i++)
	{
		for (int j = 0; j < convolutionShape[1]; j++)
		{
			xt::view(rotFilters, convolutionShape[0] - i - 1, convolutionShape[1] - j - 1, xt::all()) = xt::view(filters, i, j, xt::all());
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
	/*cout << "sigmasPrime: " << sigmasPrime.dimension() << ", "
		<< sigmasPrime.shape()[0] << " x "
		<< sigmasPrime.shape()[1] << " x "
		<< sigmasPrime.shape()[2] << " x "
		<< sigmasPrime.shape()[3] << " x "
		<< sigmasPrime.shape()[4] << endl;*/
	for (int c = 0; c < inputChannels; c++)
	{
		/*cout << "paddedSigmas: " << paddedSigmas.dimension() << ", "
			<< paddedSigmas.shape()[0] << " x "
			<< paddedSigmas.shape()[1] << " x "
			<< paddedSigmas.shape()[2] << " x "
			<< paddedSigmas.shape()[3] << " x "
			<< paddedSigmas.shape()[4] << endl;*/

		paddedWindowView[DIMC+1] = c;
		kernelWindowView[kDIMF] = c;
		sigmasPrimeWindowView[DIMC] = c;
		auto filteredSigmas = xt::xarray<double>(xt::strided_view(paddedSigmas, paddedWindowView));
		auto kernels = xt::xarray<double>(xt::strided_view(rotFilters, kernelWindowView));

		/*cout << "rotFilters: " << rotFilters.dimension() << ", "
			<< rotFilters.shape()[0] << " x "
			<< rotFilters.shape()[1] << " x "
			<< rotFilters.shape()[2] << " x "
			<< rotFilters.shape()[3] << " x "
			<< rotFilters.shape()[4] << endl;
		cout << "filteredSigmas: " << filteredSigmas.dimension() << ", "
			<< filteredSigmas.shape()[0] << " x "
			<< filteredSigmas.shape()[1] << " x "
			<< filteredSigmas.shape()[2] << " x "
			<< filteredSigmas.shape()[3] << " x "
			<< filteredSigmas.shape()[4] << endl;
		cout << "kernels: " << kernels.dimension() << ", "
			<< kernels.shape()[0] << " x "
			<< kernels.shape()[1] << " x "
			<< kernels.shape()[2] << " x "
			<< kernels.shape()[3] << " x "
			<< kernels.shape()[4] << endl;*/

		auto result = convolude(filteredSigmas, kernels);

		/*cout << "result: " << result.dimension() << ", "
			<< result.shape()[0] << " x "
			<< result.shape()[1] << " x "
			<< result.shape()[2] << " x "
			<< result.shape()[3] << " x "
			<< result.shape()[4] << endl;
		auto t = xt::strided_view(sigmasPrime, sigmasPrimeWindowView);
		cout << "t: " << t.dimension() << ", "
			<< t.shape()[0] << " x "
			<< t.shape()[1] << " x "
			<< t.shape()[2] << " x "
			<< t.shape()[3] << " x "
			<< t.shape()[4] << endl;*/

		xt::strided_view(sigmasPrime, sigmasPrimeWindowView) = result;
	}

	// Restore kernelWindowView
	kernelWindowView[kDIMF] = xt::all();

	return sigmasPrime;
}

void Convolution2DFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const int X = convolutionShape.at(0);
	const int Y = convolutionShape.at(1);

	const double RESCALE = DRAW_LEN * scale;
	double yHeight = 2.0 * RESCALE / Y;
	double xWidth = 2.0 * RESCALE / X;

	auto weights = this->getWeights().getParameters();

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
				float colorValue = ((float)(weights(i, j, 0, n)));
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