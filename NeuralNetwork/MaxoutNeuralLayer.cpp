#define _USE_MATH_DEFINES

#include "MaxoutNeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <limits>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

#include "Test.h"

using namespace std;

// Number of functions is given as k in the original paper
MaxoutNeuralLayer::MaxoutNeuralLayer(NeuralLayer* parent, size_t numUnits, size_t numFunctions, bool addBias)
{
	this->numInputs = 0;
	this->parent = parent;
	this->children = NULL;
	if (parent != NULL)
	{
		parent->addChildren(this);
		this->numInputs = parent->getNumUnits();
	}
	else { }
	this->numUnits = numUnits;
	this->numFunctions = numFunctions;

	this->addBias = addBias;
	if (addBias)
	{
		this->numInputs++;
	}
	else { }

	std::vector<size_t> paramShape;
	// input x output x functions -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	paramShape.push_back(this->numFunctions);
	this->weights.setParametersRandom(paramShape);
}

MaxoutNeuralLayer::~MaxoutNeuralLayer()
{

}

xt::xarray<double> MaxoutNeuralLayer::dotProduct(const xt::xarray<double>& input)
{
	return xt::linalg::tensordot(input, weights.getParameters(), 1); // The last dimension of the input with the first dimension of the weights
}

xt::xarray<double> MaxoutNeuralLayer::maxout(const xt::xarray<double>& input, bool storeIndices)
{
	// The input is repeated for each of the functions and then dotted with it, before finally taking the max of each of the functions
	// h_i(x) = max(W_i * x)

	// Repeat the last dimension of the input for each of the functions
	auto outputShape = input.shape();
	const size_t DIMF = outputShape.size();
	xt::xarray<double> output = xt::expand_dims(input, DIMF);
	output = xt::repeat(output, numFunctions, DIMF);

	// Perform the dot product
	// n x ... x [input -> output] x functions -shaped
	xt::xarray<double> dotput = dotProduct(input); 

	// Finally take the max value from each of the functions
	output = xt::amax(dotput, { DIMF });

	// If training, store which function was the max function for backpropagating later
	if (storeIndices)
	{
		// Add the removed dimension back in, repeat the output across it, then compare it to get a mask of which functions were taken
		xt::xarray<double> max = xt::expand_dims(output, DIMF);
		max = xt::repeat(max, numFunctions, DIMF);
		maxMask = xt::equal(max, dotput);
	}
	else { }

	return output;
}

xt::xarray<double> MaxoutNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	xt::xarray<double> output;
	if (addBias)
	{
		output = addBiasToInput(input);
	}
	else
	{
		output = input;
	}
	// Apply the maxout function
	output = maxout(output);
	return output;
}

xt::xarray<double> MaxoutNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	// Append the bias to the input and store it for backpropagating
	if (addBias)
	{
		lastInput = addBiasToInput(input);
	}
	else
	{
		lastInput = input;
	}
	// Apply the dot product and then the activation function
	lastOutput = maxout(lastInput, true);
	return lastOutput;
}

xt::xarray<double> MaxoutNeuralLayer::backPropagate(const xt::xarray<double>& sigmas)
{
	const int DIMF = lastOutput.dimension();
	
	// Repeat the sigmas across the last axis and then mask them
	xt::xarray<double> sigmasMax = xt::expand_dims(sigmas, DIMF);
	sigmasMax = xt::repeat(sigmasMax, numFunctions, DIMF);
	sigmasMax = maxMask * sigmasMax;

	// The delta weights are the last input dotted with the sigmas
	auto delta = xt::linalg::tensordot(xt::transpose(lastInput), sigmasMax, 1);
	weights.incrementDeltaParameters(-ALPHA * delta);

	// The new sigmas are the weights dotted with the sigmas
	xt::xarray<double> xWeights; // The weights (with the bias removed if bias was added)
	if (addBias)
	{
		xWeights = xt::view(weights.getParameters(), xt::range(0, lastInput.shape()[lastInput.dimension() - 1] - 1), xt::all());
	}
	else
	{
		xWeights = weights.getParameters();
	}

	auto newSigmasShape = sigmas.shape();
	newSigmasShape[newSigmasShape.size() - 1] = (addBias) ? (numInputs-1) : (numInputs); // Do not count the bias term
	xt::xarray<double> newSigmas = xt::zeros<double>(newSigmasShape);
	xt::xarray<double> maskMask = xt::zeros<double>(maxMask.shape());
	xt::xstrided_slice_vector maskMaskView;
	for (int f = 0; f < DIMF; f++)
	{
		maskMaskView.push_back(xt::all());
	}
	maskMaskView.push_back(0);
	xt::xstrided_slice_vector functionView;
	functionView.push_back(xt::all());
	functionView.push_back(xt::all());
	functionView.push_back(0);
	// For each of the functions, mask the inputs that used that function in the final output, 
	// and then dot that with that function
	for (int f = 0; f < numFunctions; f++)
	{
		xt::strided_view(maskMask, maskMaskView) = 0; // Clear the previous function
		maskMaskView[DIMF] = f;
		functionView[2] = f;
		xt::strided_view(maskMask, maskMaskView) = 1; // Set to the current function
		auto maskedSigmas = xt::sum(sigmasMax * maskMask, { DIMF } ); // Multiply by the mask to find the inputs that used this function, then reduce by a dimension
		newSigmas += xt::linalg::tensordot(maskedSigmas, xt::transpose(xt::strided_view(xWeights, functionView)), 1); // Dot the inputs with the function and add it to the sigmas
	}

	return newSigmas;
}

double MaxoutNeuralLayer::applyBackPropagate()
{
	double deltaWeight = xt::sum(xt::abs(weights.getDeltaParameters()))();
	weights.applyDeltaParameters();
	return deltaWeight; // Return the sum of how much the parameters have changed
}

std::vector<size_t> MaxoutNeuralLayer::getOutputShape()
{
	std::vector<size_t> outputShape;
	outputShape.push_back(numUnits);
	return outputShape;
}

void MaxoutNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	// Draw the neurons
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);
	}

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
			canvas->AddLine(previousNeuronPt, currentNeuronPt, GRAY, 1.0f);
		}
	} // for (int i = 0; i < numUnits; i++)

	// Draw the softmax function
	drawMaxout(canvas, origin, scale);

	if (addBias)
	{
		// Draw the bias box
		ImVec2 bTL = ImVec2(biasX, biasY);
		ImVec2 bBR = ImVec2(biasX + (BIAS_WIDTH * scale), biasY + (BIAS_HEIGHT * scale));
		biasPt = ImVec2(biasX + (BIAS_TEXT_X * scale), biasY);
		canvas->AddRectFilled(bTL, bBR, VERY_LIGHT_GRAY);
		canvas->AddRect(bTL, bBR, BLACK);
		canvas->AddText(ImGui::GetFont(), (BIAS_FONT_SIZE * scale), biasPt, BLACK, to_string(1).c_str());
	}
	else { }

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

void MaxoutNeuralLayer::drawMaxout(ImDrawList* canvas, ImVec2 origin, double scale)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);

	const double RESCALE = NeuralLayer::DRAW_LEN * scale;

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		ImVec2 start(position.x - RESCALE, position.y + RESCALE);
		ImVec2 end(position.x + RESCALE, position.y - RESCALE);

		canvas->AddRectFilled(start, end, WHITE);
		canvas->AddRect(start, end, BLACK);

		ImVec2 zero_x_left(position.x - RESCALE, position.y);
		ImVec2 zero_x_right(position.x + RESCALE, position.y);
		canvas->AddLine(zero_x_left, zero_x_right, LIGHT_GRAY);
		ImVec2 zero_y_base(position.x, position.y + RESCALE);
		ImVec2 zero_y_top(position.x, position.y - RESCALE);
		canvas->AddLine(zero_y_base, zero_y_top, LIGHT_GRAY);
	}
	
	for (int i = 0; i < numUnits; i++)
	{
		double minSlope = std::numeric_limits<float>::max();
		double maxSlope = std::numeric_limits<float>::min();
		xt::xarray<double> drawWeights = weights.getParameters();
		for (int f = 0; f < numFunctions; f++)
		{
			position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
			double slope = drawWeights(0, i, f);
			double inv_slope = 1.0 / abs(slope);
			double x1 = -min(1.0, inv_slope);
			double x2 = +min(1.0, inv_slope);
			double y1 = x1 * slope;
			double y2 = x2 * slope;

			minSlope = min(minSlope, slope);
			maxSlope = max(maxSlope, slope);

			ImVec2 l_start(position.x + (x1 * NeuralLayer::DRAW_LEN * scale), position.y - (y1 * NeuralLayer::DRAW_LEN * scale));
			ImVec2 l_end(position.x + (x2 * NeuralLayer::DRAW_LEN * scale), position.y - (y2 * NeuralLayer::DRAW_LEN * scale));

			canvas->AddLine(l_start, l_end, LIGHT_GRAY);
		} // for f

		// Redraw the max and min slopes in black
		double inv_minSlope = 1.0 / abs(minSlope);
		double x1 = -min(1.0, inv_minSlope);
		double y1 = x1 * minSlope;

		ImVec2 l_minStart(position.x + (x1 * NeuralLayer::DRAW_LEN * scale), position.y - (y1 * NeuralLayer::DRAW_LEN * scale));
		ImVec2 l_minEnd(position.x, position.y);

		canvas->AddLine(l_minStart, l_minEnd, BLACK);

		double inv_maxSlope = 1.0 / abs(maxSlope);
		double x2 = +min(1.0, inv_maxSlope);
		double y2 = x2 * maxSlope;

		ImVec2 l_maxStart(position.x, position.y);
		ImVec2 l_maxEnd(position.x + (x2 * NeuralLayer::DRAW_LEN * scale), position.y - (y2 * NeuralLayer::DRAW_LEN * scale));

		canvas->AddLine(l_maxStart, l_maxEnd, BLACK);
	} // for i
}