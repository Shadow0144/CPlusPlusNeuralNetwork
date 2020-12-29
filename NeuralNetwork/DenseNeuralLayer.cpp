#define _USE_MATH_DEFINES

#include "DenseNeuralLayer.h"

#pragma warning(push, 0)
#include <math.h>
#include <tuple>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

#include "Test.h"

DenseNeuralLayer::DenseNeuralLayer(ActivationFunctionType functionType, NeuralLayer* parent, size_t numUnits, std::map<string, double> additionalParameters, bool addBias)
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
	this->addBias = addBias;
	this->functionType = functionType;
	if (functionType == ActivationFunctionType::PReLU) // PReLU needs to know how many units
	{
		additionalParameters["numUnits"] = numUnits;
	}
	else { }
	this->activationFunction = ActivationFunctionFactory::getNewActivationFunction(functionType, additionalParameters);

	if (addBias)
	{
		this->numInputs++;
	}
	else { }

	std::vector<size_t> paramShape;
	// input x output -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits);
	this->weights.setParametersRandom(paramShape);
	
}

DenseNeuralLayer::~DenseNeuralLayer()
{
	delete activationFunction;
}

xt::xarray<double> DenseNeuralLayer::dotProduct(const xt::xarray<double>& input)
{
	return xt::linalg::tensordot(input, weights.getParameters(), 1); // The last dimension of the input with the first dimension of the weights
}

xt::xarray<double> DenseNeuralLayer::feedForward(const xt::xarray<double>& input)
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
	// Apply the dot product and then the activation function
	output = activationFunction->feedForward(dotProduct(output));
	return output;
}

xt::xarray<double> DenseNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
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
	lastOutput = activationFunction->feedForwardTrain(dotProduct(lastInput));
	return lastOutput;
}

xt::xarray<double> DenseNeuralLayer::denseBackpropagate(const xt::xarray<double>& sigmas)
{
	auto delta = xt::linalg::tensordot(xt::transpose(lastInput), sigmas, 1);

	weights.incrementDeltaParameters(-ALPHA * delta);
	auto biaslessWeights = xt::view(weights.getParameters(), xt::range(0, lastInput.shape()[lastInput.dimension() - 1] - 1), xt::all());

	auto newSigmas = xt::linalg::tensordot(sigmas, xt::transpose(biaslessWeights), 1); // The last axis of errors and the first axis of the transposed weights

	return newSigmas;
}

xt::xarray<double> DenseNeuralLayer::backPropagate(const xt::xarray<double>& sigmas)
{
	return denseBackpropagate(activationFunction->getGradient(sigmas));
}

double DenseNeuralLayer::applyBackPropagate()
{
	double deltaWeight = xt::sum(xt::abs(weights.getDeltaParameters()))();
	weights.applyDeltaParameters(); // Update the weights
	activationFunction->applyBackPropagate(); // Update any parameters the activation function needs to change
	return deltaWeight; // Return the sum of how much the parameters have changed
}

std::vector<size_t> DenseNeuralLayer::getOutputShape()
{
	std::vector<size_t> outputShape;
	outputShape.push_back(numUnits);
	outputShape = activationFunction->getOutputShape(outputShape);
	return outputShape;
}

void DenseNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	// Draw the neurons
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);
	}

	// Draw the activation function (If the activation function type is Identity, the Linear function will still draw)
	activationFunction->draw(canvas, origin, scale, numUnits, weights);

	// Draw the links to the previous neurons
	double previousX, previousY;
	int parentCount = parent->getNumUnits();
	const double PARENT_LAYER_WIDTH = NeuralLayer::getLayerWidth(parentCount, scale);
	ImVec2 currentNeuronPt(0, origin.y - (RADIUS * scale));
	previousY = origin.y - (DIAMETER * scale);

	xt::xarray<double> drawWeights = weights.getParameters();

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
			ImColor lineColor = ImColor(1.0f, 1.0f, 1.0f, 1.0f);
			float lineWidth = (1.0f / 36.0f) * scale;
			float weight = drawWeights(i);
			if (weight >= 0.0f)
			{
				if (weight <= 1.0f)
				{
					lineColor = ImColor(1.0f - weight, 1.0f - weight, 1.0f - weight, 1.0f);
					lineWidth = weight * lineWidth;
				}
				else
				{
					lineColor = ImColor(0.0f, 0.0f, 0.0f, 1.0f);
					lineWidth = 1.0f * lineWidth;
				}
			}
			else
			{
				if (weight >= -1.0f)
				{
					lineColor = ImColor(-weight, 0.0f, 0.0f, 1.0f);
					lineWidth = -weight * lineWidth;
				}
				else
				{
					lineColor = ImColor(1.0f, 0.0f, 0.0f, 1.0f);
					lineWidth = 1.0f * lineWidth;
				}
			}

			// Draw line
			if (lineWidth > 0)
			{
				canvas->AddLine(previousNeuronPt, currentNeuronPt, lineColor, lineWidth);
			}
			else
			{
				canvas->AddLine(previousNeuronPt, currentNeuronPt, WHITE, 1.0f);
			}
		}

		// Consider moving to another function for second pass
		if (addBias)
		{
			// Draw the bias line
			ImColor lineColor = ImColor(1.0f, 1.0f, 1.0f, 1.0f);
			float lineWidth = (1.0f / 36.0f) * scale;
			float weight = ((float)(drawWeights(numUnits - 1)));
			if (weight >= 0.0f)
			{
				if (weight <= 1.0f)
				{
					lineColor = ImColor(1.0f - weight, 1.0f - weight, 1.0f - weight, 1.0f);
					lineWidth = weight * lineWidth;
				}
				else
				{
					lineColor = ImColor(0.0f, 0.0f, 0.0f, 1.0f);
					lineWidth = 1.0f * lineWidth;
				}
			}
			else
			{
				if (weight >= -1.0f)
				{
					lineColor = ImColor(-weight, 0.0f, 0.0f, 1.0f);
					lineWidth = -weight * lineWidth;
				}
				else
				{
					lineColor = ImColor(1.0f, 0.0f, 0.0f, 1.0f);
					lineWidth = 1.0f * lineWidth;
				}
			}
			if (lineWidth > 0)
			{
				canvas->AddLine(biasPt, currentNeuronPt, lineColor, lineWidth);
			}
			else
			{
				canvas->AddLine(biasPt, currentNeuronPt, WHITE, 1.0f);
			}
		}
		else { }
	} // for (int i = 0; i < numUnits; i++)

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