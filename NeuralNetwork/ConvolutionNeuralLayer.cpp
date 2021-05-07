#include "ConvolutionNeuralLayer.h"

#include "NeuralNetworkFileHelper.h"
#include "ActivationFunctionFactory.h"
#include "NetworkExceptions.h"

#pragma warning(push, 0)
#include <xtensor/xnpy.hpp>
#pragma warning(pop)

using namespace std;

ConvolutionNeuralLayer::ConvolutionNeuralLayer(NeuralLayer* parent, size_t dims, size_t numKernels,
	const std::vector<size_t>& convolutionShape, const std::vector<size_t>& stride, bool addBias,
	ActivationFunctionType activationFunctionType, std::map<string, double> additionalParameters)
	: ParameterizedNeuralLayer(parent)
{
	this->numUnits = numKernels;
	this->convolutionShape = convolutionShape;
	this->stride = stride;
	this->numKernels = numKernels;
	this->activationFunction = ActivationFunctionFactory::getNewActivationFunction(activationFunctionType, additionalParameters);

	if (convolutionShape.size() == 1)
	{
		for (int i = 1; i < dims; i++) // Start at 1
		{
			this->convolutionShape.push_back(convolutionShape[0]);
		}
	}
	else if (convolutionShape.size() != dims)
	{
		throw NeuralLayerConvolutionShapeException();
	}

	auto inputShape = parent->getOutputShape();
	const int iDIMS = inputShape.size();
	if (iDIMS < (dims + 1)) // Make sure there are enough dimensions in the input
	{
		throw NeuralLayerInputShapeException();
	}
	else { }
	this->inputChannels = inputShape.at(iDIMS - 1);
	for (int i = 0; i < dims; i++) // Make sure the input is at least as large as the convolution
	{
		if (inputShape[iDIMS - i - 2] < this->convolutionShape[dims - i - 1]) // - 2 because of the channels
		{
			throw NeuralLayerConvolutionShapeException();
		}
	}

	if (stride.size() == 1)
	{
		for (int i = 1; i < dims; i++) // Start at 1
		{
			this->stride.push_back(stride[0]);
		}
	}
	else if (stride.size() != dims)
	{
		throw NeuralLayerStrideShapeException();
	}

	std::vector<size_t> paramShape;
	// Convolution x ... x filters x kernel -shaped
	kernelWindowView = xt::xstrided_slice_vector();

	for (int i = 0; i < dims; i++)
	{
		paramShape.push_back(this->convolutionShape[i]);
		kernelWindowView.push_back(xt::all());
	}

	kernelWindowView.push_back(xt::all()); // Channels / Filters
	kernelWindowView.push_back(0); // Current kernel
	paramShape.push_back(inputChannels);
	paramShape.push_back(numKernels);
	this->weights.setParametersRandom(paramShape);

	if (hasBias)
	{
		this->biasWeights.setParametersRandom(numKernels);
		this->biasWeights.setUnregularized(); // Bias is typically unregularized
	}
	else { }
}

ConvolutionNeuralLayer::~ConvolutionNeuralLayer()
{

}

xt::xarray<double> ConvolutionNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	return activationFunction->feedForward(convolveInput(input));
}

xt::xarray<double> ConvolutionNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	lastInput = input;
	lastOutput = activationFunction->feedForwardTrain(convolveInput(input));
	return lastOutput;
}

double ConvolutionNeuralLayer::applyBackPropagate()
{
	double deltaWeight = xt::sum(xt::abs(weights.getDeltaParameters()))();
	weights.applyDeltaParameters();
	if (hasBias)
	{
		deltaWeight += xt::sum(xt::abs(biasWeights.getDeltaParameters()))();
		biasWeights.applyDeltaParameters();
	}
	else { }
	deltaWeight += activationFunction->applyBackPropagate();
	return deltaWeight; // Return the sum of how much the parameters have changed
}

void ConvolutionNeuralLayer::saveParameters(std::string fileName)
{
	ParameterizedNeuralLayer::saveParameters(fileName); // Handles the wieghts and activation function
	if (hasBias)
	{
		xt::dump_npy(fileName + "_b.npy", biasWeights.getParameters());
	}
	else { }
}

void ConvolutionNeuralLayer::loadParameters(std::string fileName)
{
	ParameterizedNeuralLayer::loadParameters(fileName); // Handles the weights and activation function
	if (hasBias)
	{
		bool exists = NeuralNetworkFileHelper::fileExists(fileName + "_b.npy");
		if (exists)
		{
			weights.setParameters(xt::load_npy<double>(fileName + "_b.npy"));
		}
		else
		{
			cout << "Parameter file " + fileName + "_b.npy" + " not found" << endl;
		}
	}
	else { }
}

std::vector<size_t> ConvolutionNeuralLayer::getOutputShape()
{
	auto shape = parent->getOutputShape();
	const int S = shape.size();
	const int C = convolutionShape.size();
	if (S < (C + 1)) // Plus channels
	{
		throw NeuralLayerInputShapeException();
	}
	else { }
	for (int i = 0; i < C; i++) // Convoluded dimensions
	{
		// ceil((shape[DIM1] - (convolutionShape[0] - 1)) / stride[0])
		shape[S - C + i - 1] = ceil((shape[S - C + i - 1] - (convolutionShape[i] - 1)) / stride[i]);
	}
	// Channel dimension
	shape[S - 1] = numKernels;
	return shape;
}

void ConvolutionNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
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
	drawConvolution(canvas, origin, scale);

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