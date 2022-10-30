#include "SqueezeNeuralLayer.h"

#include "NetworkExceptions.h"

// The squeeze dims account for only the shape of a single example
SqueezeNeuralLayer::SqueezeNeuralLayer(NeuralLayer* parent, const std::vector<size_t>& squeezeDims)
	: ShapeNeuralLayer(parent)
{
	this->squeezeDims = squeezeDims;
}

SqueezeNeuralLayer::~SqueezeNeuralLayer()
{

}

xt::xarray<double> SqueezeNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	auto shape = input.shape();
	std::vector<size_t> fullNewShape = { shape[0] };
	const int DIMS = shape.size();
	const int sDIMS = squeezeDims.size();
	if (sDIMS != 0) // If passed a non-empty array for the dimensions to squeeze, squeeze them
	{
		int i = 1;
		for (int s = 0; s < sDIMS;) // The length of squeezeDims should be no longer than the input, so loop on that first
		{
			if (i != (squeezeDims[s] + 1)) // If the next dimension is not to be squeezed, add it
			{
				fullNewShape.push_back(shape[i]);
			}
			else // Otherwise, look at the next dimension to squeeze
			{ 
				s++;
			}
			i++; // Move to the next input dimension each iteration
		}
		for (; i < DIMS; i++) // Add any remaining dimensions
		{
			fullNewShape.push_back(shape[i]);
		}
	}
	else // If passed an empty array for the dimensions to squeeze, squeeze all dimensions after the first of size 1
	{
		for (int i = 1; i < DIMS; i++)
		{
			if (shape[i] != 1 && shape[i] != 0) // Add any dimensions which are not of size 0 or 1
			{
				fullNewShape.push_back(shape[i]);
			}
		}
	}
	auto result = xt::xarray<double>(input); // Reshape the input
	result.reshape(fullNewShape);
	return result;
}

std::vector<size_t> SqueezeNeuralLayer::getOutputShape()
{
	auto shape = parent->getOutputShape();
	const int DIMS = shape.size();
	const int sDIMS = squeezeDims.size();
	if (sDIMS > DIMS)
	{
		throw NeuralLayerSqueezeShapeException();
	}
	else { }
	std::vector<size_t> result;
	if (sDIMS == 0) // If the list is empty, remove all dimensions of size 0 or 1
	{
		for (int i = 0; i < DIMS; i++)
		{
			if (shape[i] != 0 && shape[i] != 1) // Push back any non-0 and non-1 dimensions
			{
				result.push_back(shape[i]);
			}
			else { }
		}
	}
	else // If the list is not empty, remove all dimensions specified
	{
		int s = 0;
		for (int i = 0; i < DIMS; i++)
		{
			if ((s < sDIMS) && (i == squeezeDims[s])) // Check if a dimension is on the list
			{
				if (shape[i] != 0 && shape[i] != 1) // If the removed dimension is not size 0 or 1, throw an exception
				{
					throw NeuralLayerSqueezeShapeException();
				}
				else { }
				s++; // Skip adding this dimension
			}
			else // If not on the list, add it
			{
				result.push_back(shape[i]);
			}
		}
	}
	return result;
}

void SqueezeNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
{
	// Draw the neuron
	ImVec2 position = ImVec2(origin);
	const double LAYER_WIDTH = getLayerWidth(1, scale);
	position.x = getNeuronX(origin.x, LAYER_WIDTH, 0, scale);
	canvas->AddCircleFilled(position, RADIUS * scale, LIGHT_GRAY, 32);

	// Draw the activation function
	drawSqueezeFunction(canvas, origin, scale);

	// Draw the links to the previous neurons
	double previousX, previousY;
	int parentCount = parent->getNumUnits();
	const double PARENT_LAYER_WIDTH = NeuralLayer::getLayerWidth(parentCount, scale);
	ImVec2 currentNeuronPt(0, origin.y - (RADIUS * scale));
	previousY = origin.y - (DIAMETER * scale);

	// Draw the neuron
	currentNeuronPt.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, 0, scale);
	for (int j = 0; j < parentCount; j++) // There should be at least one parent
	{
		previousX = NeuralLayer::getNeuronX(origin.x, PARENT_LAYER_WIDTH, j, scale);
		ImVec2 previousNeuronPt(previousX, previousY);

		// Draw line to previous neuron
		canvas->AddLine(previousNeuronPt, currentNeuronPt, BLACK, 1.0f);
	}

	if (output)
	{
		// Draw the output lines
		ImVec2 outputPt(position.x, position.y + (RADIUS * scale));
		ImVec2 nextPt(position.x, outputPt.y + (LINE_LENGTH * scale));
		canvas->AddLine(outputPt, nextPt, GRAY);
	}
	else { }

	// Overlaying black ring
	canvas->AddCircle(position, RADIUS * scale, BLACK, 32);
}

void SqueezeNeuralLayer::drawSqueezeFunction(ImDrawList* canvas, ImVec2 origin, double scale)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);

	const double ARROW_WIDTH = 2.0 * scale;
	const double ARROW_HEIGHT = 4.0 * scale;
	const double ARROW_OVERLAP = 0.1 * scale;

	const double RESCALE = DRAW_LEN * scale * RERESCALE;

	const double FONT_SIZE = 11.0 * scale;
	const double LEFT_X = 24.0 * scale;
	const double LEFT_Y = 27.5 * scale;
	const double RIGHT_X = 6.0 * scale;
	const double RIGHT_Y = 16.5 * scale;

	// There is only one "neuron"
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(1, scale);
	origin.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, 0, scale);

	ImVec2 position(origin.x, origin.y);

	// Draw arrow
	canvas->AddLine(ImVec2(position.x + ARROW_WIDTH, position.y + ARROW_OVERLAP),
		ImVec2(position.x - ARROW_WIDTH, position.y - ARROW_HEIGHT), BLACK);
	canvas->AddLine(ImVec2(position.x + ARROW_WIDTH, position.y - ARROW_OVERLAP),
		ImVec2(position.x - ARROW_WIDTH, position.y + ARROW_HEIGHT), BLACK);

	// Get font
	ImGuiIO& io = ImGui::GetIO();
	ImFont* font = io.Fonts->Fonts[0];

	//Draw left
	ImVec2 start(position.x - LEFT_X, position.y - LEFT_Y);
	canvas->AddText(font, FONT_SIZE, start, BLACK, "[x]\n[1]\n[y]\n[1]\n[z]");

	// Draw right
	start = ImVec2(position.x + RIGHT_X, position.y - RIGHT_Y);
	canvas->AddText(font, FONT_SIZE, start, BLACK, "[x]\n[y]\n[z]");
}