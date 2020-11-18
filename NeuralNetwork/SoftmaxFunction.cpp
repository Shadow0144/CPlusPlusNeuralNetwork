#include "SoftmaxFunction.h"
#include "NeuralLayer.h"

#include "Test.h"

#pragma warning(push, 0)
#include <iostream>
#include <cmath>
#include <mutex>  // For std::unique_lock
#pragma warning(pop)

using namespace std;

SoftmaxFunction::SoftmaxFunction(size_t incomingUnits, int axis)
{
	this->hasBias = false;
	this->numInputs = incomingUnits;
	this->numUnits = 1;
	this->numOutputs = numInputs;
	this->axis = axis;
	this->drawAxes = false;
}

xt::xarray<double> SoftmaxFunction::feedForward(xt::xarray<double> inputs)
{
	int sumAxis = (axis > 0) ? (axis) : (inputs.dimension() + axis);

	double c = 0;// -0.1; // negative max per axis // TODO
	auto z = xt::exp(inputs + c);

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

xt::xarray<double> SoftmaxFunction::feedForwardTrain(xt::xarray<double> inputs)
{
	lastInput = inputs; // No bias
	outputMutex.lock();
	lastOutput = feedForward(lastInput);
	outputMutex.unlock();
	return lastOutput;
}

xt::xarray<double> SoftmaxFunction::backPropagate(xt::xarray<double> sigmas)
{
	auto newSigmas = xt::pow(sigmas, 2.0); // TODO? Potentially wrong equation
	return newSigmas;
}

xt::xarray<double> SoftmaxFunction::backPropagateCrossEntropy(xt::xarray<double> sigmas)
{
	return sigmas;
}

void SoftmaxFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor GRAY(0.3f, 0.3f, 0.3f, 1.0f);
	const ImColor LIGHT_GRAY(0.6f, 0.6f, 0.6f, 1.0f);

	const double RESCALE = DRAW_LEN * scale;
	double yHeight = 2.0 * RESCALE;
	double xWidth = 2.0 * RESCALE / numOutputs;

	outputMutex.lock_shared();
	xt::xarray<double> output = xt::xarray<double>(lastOutput);
	outputMutex.unlock();

	auto shape = output.shape(); // Show the first example
	xt::xstrided_slice_vector sv;
	int dims = output.dimension();
	int stop = dims - 1;
	for (int i = 0; i < stop; i++)
	{
		//sv.push_back(shape[i] - 1);
		sv.push_back(0);
	}
	sv.push_back(xt::all());

	ImVec2 position(0, origin.y + RESCALE);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);

		sv[dims - 2] = i; // Select correct neuron
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