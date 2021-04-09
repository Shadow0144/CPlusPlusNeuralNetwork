#include "NeuralLayer.h"
#include "ActivationFunction.h"

#include "Test.h"

const double NeuralLayer::DRAW_LEN = 16.0;
const double NeuralLayer::RERESCALE = 0.75;
const double NeuralLayer::SHIFT = 16.0;
const double NeuralLayer::RADIUS = 40;
const double NeuralLayer::DIAMETER = RADIUS * 2;
const double NeuralLayer::NEURON_SPACING = 20;
const double NeuralLayer::LINE_LENGTH = 15;
const double NeuralLayer::WEIGHT_RADIUS = 10;
const double NeuralLayer::BIAS_OFFSET_X = 40;
const double NeuralLayer::BIAS_OFFSET_Y = -52;
const double NeuralLayer::BIAS_FONT_SIZE = 24;
const double NeuralLayer::BIAS_WIDTH = 20;
const double NeuralLayer::BIAS_HEIGHT = BIAS_FONT_SIZE;
const double NeuralLayer::BIAS_TEXT_X = 4;
const double NeuralLayer::BIAS_TEXT_Y = 20;
const ImColor NeuralLayer::BLACK = ImColor(0.0f, 0.0f, 0.0f, 1.0f);
const ImColor NeuralLayer::GRAY = ImColor(0.3f, 0.3f, 0.3f, 1.0f);
const ImColor NeuralLayer::LIGHT_GRAY = ImColor(0.6f, 0.6f, 0.6f, 1.0f);
const ImColor NeuralLayer::VERY_LIGHT_GRAY = ImColor(0.8f, 0.8f, 0.8f, 1.0f);
const ImColor NeuralLayer::WHITE = ImColor(1.0f, 1.0f, 1.0f, 1.0f);

using namespace std;

NeuralLayer::~NeuralLayer()
{
	if (activationFunction != nullptr)
	{
		delete activationFunction;
	}
	else { }
}

void NeuralLayer::addChildren(NeuralLayer* children)
{
	this->children = children;
}

size_t NeuralLayer::getNumUnits() 
{ 
	return numUnits;
}

xt::xarray<double> NeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	lastInput = input;
	lastOutput = feedForward(input);
	return lastOutput;
}

std::vector<size_t> NeuralLayer::getOutputShape()
{
	std::vector<size_t> outputShape;
	outputShape.push_back(numUnits);
	return outputShape;
}

double NeuralLayer::getRegularizationLoss(double lambda1, double lambda2) const
{
	double loss = 0.0;
	if (activationFunction != nullptr)
	{
		loss += activationFunction->getRegularizationLoss(lambda1, lambda2);
	}
	else { }
	return loss;
}

xt::xarray<double> NeuralLayer::addBiasToInput(const xt::xarray<double>& input)
{
	size_t inputDims = input.dimension();
	auto inputShape = input.shape();
	inputShape.at(inputDims - 1)++; // Add one to the last 
	xt::xstrided_slice_vector biaslessView;
	for (int i = 0; i < (inputDims - 1); i++)
	{
		biaslessView.push_back(xt::all());
	}
	biaslessView.push_back(xt::range(0, (inputShape.at(inputDims - 1) - 1)));

	xt::xarray<double> biasedInput = xt::ones<double>(inputShape);
	xt::strided_view(biasedInput, biaslessView) = input;

	return biasedInput;
}

void NeuralLayer::saveParameters(std::string fileName)
{
	if (activationFunction != nullptr)
	{
		activationFunction->saveParameters(fileName);
	}
	else { }
}

void NeuralLayer::loadParameters(std::string fileName)
{
	if (activationFunction != nullptr)
	{
		activationFunction->loadParameters(fileName);
	}
	else { }
}

void NeuralLayer::substituteParameters(Optimizer* optimizer)
{
	if (activationFunction != nullptr)
	{
		activationFunction->substituteParameters(optimizer);
	}
	else { }
}

void NeuralLayer::restoreParameters(Optimizer* optimizer)
{
	if (activationFunction != nullptr)
	{
		activationFunction->restoreParameters(optimizer);
	}
	else { }
}

double NeuralLayer::getLayerWidth(size_t numUnits, double scale)
{
	return ((((DIAMETER + NEURON_SPACING) * numUnits) + NEURON_SPACING) * scale);
}

double NeuralLayer::getNeuronX(double originX, double layerWidth, int i, double scale)
{
	return (originX - (layerWidth * 0.5) + (((DIAMETER + NEURON_SPACING) * i) * scale));
}

void NeuralLayer::drawFunctionBackground(ImDrawList* canvas, ImVec2 origin, double scale, bool drawAxes)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);

	const double RESCALE = DRAW_LEN * scale;

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		ImVec2 start(position.x - RESCALE, position.y + RESCALE);
		ImVec2 end(position.x + RESCALE, position.y - RESCALE);

		canvas->AddRectFilled(start, end, WHITE);
		canvas->AddRect(start, end, BLACK);

		if (drawAxes)
		{
			ImVec2 zero_x_left(position.x - RESCALE, position.y);
			ImVec2 zero_x_right(position.x + RESCALE, position.y);
			canvas->AddLine(zero_x_left, zero_x_right, LIGHT_GRAY);
			ImVec2 zero_y_base(position.x, position.y + RESCALE);
			ImVec2 zero_y_top(position.x, position.y - RESCALE);
			canvas->AddLine(zero_y_base, zero_y_top, LIGHT_GRAY);
		}
		else { }
	}
}

void NeuralLayer::drawConversionFunctionBackground(ImDrawList* canvas, ImVec2 origin, double scale, bool drawAxes)
{
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);
	const ImColor WHITE(1.0f, 1.0f, 1.0f, 1.0f);

	const double ARROW_WIDTH = 2.0 * scale;
	const double ARROW_HEIGHT = 4.0 * scale;

	const double RESCALE = DRAW_LEN * scale * RERESCALE;

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		// Draw arrow
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		canvas->AddLine(ImVec2(position.x + ARROW_WIDTH, position.y),
			ImVec2(position.x - ARROW_WIDTH, position.y - ARROW_HEIGHT), BLACK);
		canvas->AddLine(ImVec2(position.x + ARROW_WIDTH, position.y),
			ImVec2(position.x - ARROW_WIDTH, position.y + ARROW_HEIGHT), BLACK);

		//Draw left
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale) - (SHIFT * scale);
		ImVec2 start(position.x - RESCALE, position.y + RESCALE);
		ImVec2 end(position.x + RESCALE, position.y - RESCALE);

		canvas->AddRectFilled(start, end, WHITE);
		canvas->AddRect(start, end, BLACK);

		if (drawAxes)
		{
			ImVec2 zero_x_left(position.x - RESCALE, position.y);
			ImVec2 zero_x_right(position.x + RESCALE, position.y);
			canvas->AddLine(zero_x_left, zero_x_right, LIGHT_GRAY);
			ImVec2 zero_y_base(position.x, position.y + RESCALE);
			ImVec2 zero_y_top(position.x, position.y - RESCALE);
			canvas->AddLine(zero_y_base, zero_y_top, LIGHT_GRAY);
		}
		else { }

		// Draw right
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale) + (SHIFT * scale);
		start = ImVec2(position.x - RESCALE, position.y + RESCALE);
		end = ImVec2(position.x + RESCALE, position.y - RESCALE);

		canvas->AddRectFilled(start, end, WHITE);
		canvas->AddRect(start, end, BLACK);

		if (drawAxes)
		{
			ImVec2 zero_x_left(position.x - RESCALE, position.y);
			ImVec2 zero_x_right(position.x + RESCALE, position.y);
			canvas->AddLine(zero_x_left, zero_x_right, LIGHT_GRAY);
			ImVec2 zero_y_base(position.x, position.y + RESCALE);
			ImVec2 zero_y_top(position.x, position.y - RESCALE);
			canvas->AddLine(zero_y_base, zero_y_top, LIGHT_GRAY);
		}
		else { }
	}
}