#define _USE_MATH_DEFINES

#include "NeuralLayer/SoftmaxNeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <cmath>
#include <mutex>  // For std::unique_lock
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

#include "Test.h"

using namespace std;

SoftmaxNeuralLayer::SoftmaxNeuralLayer(NeuralLayer* parent, int axis)
	: NeuralLayer(parent)
{
	// TODO: Assert parent dimension
	this->numUnits = 1;
	this->numOutputs = parent->getOutputShape()[parent->getOutputShape().size()-1];
	this->numInputs = numOutputs;

	this->axis = axis;

	this->useOptimizedGradient = false;
}

SoftmaxNeuralLayer::~SoftmaxNeuralLayer()
{
	
}

xt::xarray<double> SoftmaxNeuralLayer::feedForward(const xt::xarray<double>& input)
{
	int sumAxis = (axis > 0) ? (axis) : (input.dimension() + axis);

	double c = 0;// -0.1; // negative max per axis // TODO
	auto z = xt::exp(input + c);

	// We lose a dimension when summing, so broadcasting won't work without this trick
	auto shape = z.shape();
	shape[sumAxis] = 1;
	xt::xstrided_slice_vector dimensionView;
	for (int i = 0; i < sumAxis; i++)
	{
		dimensionView.push_back(xt::all());
	}
	dimensionView.push_back(0);
	dimensionView.push_back(xt::ellipsis());
	xt::xarray<double> total(shape);
	xt::strided_view(total, dimensionView) = xt::sum<double>(z, { sumAxis });

	xt::xarray<double> output = z / total;
	output = xt::nan_to_num(output);

	return output;
}

xt::xarray<double> SoftmaxNeuralLayer::feedForwardTrain(const xt::xarray<double>& input)
{
	lastInput = input; // No bias
	outputMutex.lock();
	lastOutput = feedForward(lastInput);
	outputMutex.unlock();
	return lastOutput;
}

xt::xarray<double> SoftmaxNeuralLayer::getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	xt::xarray<double> newSigmas;

	if (useOptimizedGradient)
	{
		newSigmas = getGradientCrossEntropy(sigmas, optimizer);
	}
	else
	{
		newSigmas = getGradientStandard(sigmas, optimizer);
	}

	return newSigmas;
}

xt::xarray<double> SoftmaxNeuralLayer::getGradientStandard(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	xt::xarray<double> newSigmas = xt::zeros<double>(sigmas.shape());

	xt::xstrided_slice_vector exampleView;
	exampleView.push_back(0);
	exampleView.push_back(xt::ellipsis());

	const int N = sigmas.shape()[sigmas.shape().size() - 1];
	for (int n = 0; n < N; n++)
	{
		auto sigma = xt::strided_view(sigmas, exampleView);
		auto output = xt::strided_view(lastOutput, exampleView);
		auto gradient = xt::diag(output) - xt::linalg::outer(output, output);
		xt::strided_view(newSigmas, exampleView) = xt::linalg::tensordot(sigma, gradient, 1);
	}

	return newSigmas;
}

xt::xarray<double> SoftmaxNeuralLayer::getGradientCrossEntropy(const xt::xarray<double>& sigmas, Optimizer* optimizer)
{
	return sigmas;
}

double SoftmaxNeuralLayer::applyBackPropagate()
{
	return 0; // No parameters
}

std::vector<size_t> SoftmaxNeuralLayer::getOutputShape()
{
	return parent->getOutputShape();
}

void SoftmaxNeuralLayer::useSimplifiedGradient(bool useOptimizedGradient)
{
	this->useOptimizedGradient = useOptimizedGradient;
}

bool SoftmaxNeuralLayer::isSoftmaxLayer()
{
	return true;
}

void SoftmaxNeuralLayer::draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output)
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
	drawSoftmax(canvas, origin, scale);

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

void SoftmaxNeuralLayer::drawSoftmax(ImDrawList* canvas, ImVec2 origin, double scale)
{
	drawFunctionBackground(canvas, origin, scale, false);

	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);

	const double RESCALE = DRAW_LEN * scale;
	double yHeight = 2.0 * RESCALE;
	double xWidth = 2.0 * RESCALE / numOutputs;

	outputMutex.lock_shared();
	xt::xarray<double> output = xt::xarray<double>(lastOutput);
	outputMutex.unlock_shared();

	const int DIMS = output.dimension();
	if (DIMS > 2) // Multiple neurons
	{
		xt::xstrided_slice_vector sv;
		const int STOP = DIMS - 1; // Subtract the output dimension
		for (int i = 0; i < STOP; i++)
		{
			sv.push_back(0);
		}
		sv.push_back(xt::all());

		ImVec2 position(0, origin.y + RESCALE);
		const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
		for (int i = 0; i < numUnits; i++)
		{
			position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

			sv[DIMS - 2] = i; // Select correct neuron
			double x = -RESCALE;
			for (int j = 0; j < numOutputs; j++)
			{
				ImColor color = (j % 2 == 0) ? GRAY : LIGHT_GRAY;
				double y = xt::strided_view(output, sv)(j) * RESCALE * 2; // Select correct output and calculate scale
				canvas->AddRectFilled(ImVec2(position.x + x, position.y), ImVec2(position.x + x + xWidth, position.y - y), color);
				x += xWidth;
			}
		}
	}
	else if (DIMS > 1) // Only one neuron
	{
		xt::xstrided_slice_vector sv;
		const int STOP = DIMS - 1; // Subtract the output dimension
		for (int i = 0; i < STOP; i++)
		{
			sv.push_back(0);
		}
		sv.push_back(xt::all());

		ImVec2 position(0, origin.y + RESCALE);
		const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, 0, scale);
		double x = -RESCALE;
		for (int j = 0; j < numOutputs; j++)
		{
			ImColor color = (j % 2 == 0) ? GRAY : LIGHT_GRAY;
			double y = xt::strided_view(output, sv)(j) * RESCALE * 2; // Select correct output and calculate scale
			canvas->AddRectFilled(ImVec2(position.x + x, position.y), ImVec2(position.x + x + xWidth, position.y - y), color);
			x += xWidth;
		}
	}
	else { } // TODO - Draw something empty
}