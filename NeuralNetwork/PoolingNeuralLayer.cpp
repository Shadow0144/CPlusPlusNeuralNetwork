#include "PoolingNeuralLayer.h"

#include "NetworkExceptions.h"

PoolingNeuralLayer::PoolingNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& filterShape, bool hasChannels)
	: ParameterizedNeuralLayer(parent)
{
	this->filterShape = filterShape;
	this->hasChannels = hasChannels;
	this->numUnits = 1;
}

std::vector<size_t> PoolingNeuralLayer::getOutputShape()
{
	auto shape = parent->getOutputShape();
	const int S = shape.size();
	const int C = filterShape.size();
	if (hasChannels)
	{
		if (S < (C + 1)) // Plus channels
		{
			throw NeuralLayerInputShapeException();
		}
		else { }
		for (int i = 0; i < C; i++) // Pooled dimensions
		{
			shape[S - C + i - 1] = shape[S - C + i - 1] / filterShape[i];
		}
		// Channels remain the same
	}
	else 
	{
		if (S < C) // No channels
		{
			throw NeuralLayerInputShapeException();
		}
		else { }
		for (int i = 0; i < C; i++) // Pooled dimensions
		{
			shape[S - C + i] = shape[S - C + i] / filterShape[i];
		}
	}
	return shape;
}

void PoolingNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor VERY_LIGHT_GRAY(0.8f, 0.8f, 0.8f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);
	const double LINE_LENGTH = 15;

	// Draw the neurons
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);
	}

	// Draw the pooling function
	drawPooling(canvas, origin, scale);

	// Draw the links to the previous neurons
	double previousX, previousY;
	int parentCount = parent->getNumUnits();
	const double PARENT_LAYER_WIDTH = NeuralLayer::getLayerWidth(parentCount, scale);
	ImVec2 currentNeuronPt(0, origin.y - (RADIUS * scale));
	previousY = origin.y - (DIAMETER * scale);

	// Draw each neuron
	for (int i = 0; i < numUnits; i++)
	{
		currentNeuronPt.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		for (int j = 0; j < parentCount; j++) // There should be at least one parent
		{
			previousX = NeuralLayer::getNeuronX(origin.x, PARENT_LAYER_WIDTH, j, scale);
			ImVec2 previousNeuronPt(previousX, previousY);

			// Draw line to previous neuron
			canvas->AddLine(previousNeuronPt, currentNeuronPt, BLACK, 1.0f);
		}
	}

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