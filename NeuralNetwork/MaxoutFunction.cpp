#include "MaxoutFunction.h"
#include "NeuralLayer.h"

#pragma warning(push, 0)
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#pragma warning(pop)

using namespace std;

// A maxout unit takes the maximum value among the values from n linear functions
MaxoutFunction::MaxoutFunction(size_t incomingUnits, size_t numUnits, size_t numFunctions)
{
	this->hasBias = true;
	this->numUnits = numUnits;
	this->numFunctions = numFunctions;
	this->numInputs = incomingUnits + 1; // Plus bias
	std::vector<size_t> paramShape;
	// input x (output x functions) -shaped
	paramShape.push_back(this->numInputs);
	paramShape.push_back(this->numUnits * this->numFunctions);
	this->weights.setParametersRandom(paramShape);
}

xt::xarray<double> MaxoutFunction::feedForward(xt::xarray<double> inputs)
{
	// h_i(x) = max(W_i * x)
	lastInput = inputs;
	lastOutput = dotProduct(inputs);

	// n x ... x [input -> output] x functions -shaped
	auto outputShape = inputs.shape();
	int features = outputShape.size();
	outputShape[features-1] = numUnits;
	outputShape.push_back(numFunctions);

	// Reduce the last dimension to produce one output per neuron
	lastOutput.reshape(outputShape);
	lastIndices = xt::argmax(lastOutput, { features });
	lastOutput = xt::amax(lastOutput, { features });
	return lastOutput;
}

xt::xarray<double> MaxoutFunction::backPropagate(xt::xarray<double> sigmas)
{
	int N = lastInput.shape()[0];
	std::vector<size_t> primeShape;
	primeShape.push_back(N);
	primeShape.push_back(numUnits * numFunctions);
	// N x (output x functions) -shaped
	xt::xarray<double> sigmasPrime = xt::zeros<double>(primeShape);
	for (int n = 0; n < N; n++)
	{
		for (int o = 0; o < numUnits; o++)
		{
			int index = (o * numFunctions) + lastIndices(n, o);
			sigmasPrime(n, index) = sigmas(n, o);
		}
	}

	auto delta = xt::linalg::tensordot(xt::transpose(lastInput), sigmasPrime, 1);

	weights.incrementDeltaParameters(-ALPHA * delta);
	auto biaslessWeights = xt::view(weights.getParameters(), xt::range(0, (numInputs - 1)), xt::all());

	auto newSigmas = xt::linalg::tensordot(sigmasPrime, xt::transpose(biaslessWeights), 1); // The last {1} axes of errors and the first {1} axes of the weights transposed

	return newSigmas;
}

xt::xarray<double> MaxoutFunction::activationDerivative()
{
	int dims = lastIndices.dimension();
	int functionDim = dims - 2;
	xt::xstrided_slice_vector maskedView;
	for (int i = 0; i < functionDim; i++)
	{
		maskedView.push_back(xt::all());
	}
	xt::xarray<double> mask = xt::zeros<double>(weights.getParameters().shape());
	for (int i = 0; i < numFunctions; i++)
	{
		for (int i = 0; i < numUnits; i++)
		{
			xt::strided_view(mask, maskedView)(numFunctions, numUnits) = 1;
		}
	}

	return xt::xarray<double>();
}

void MaxoutFunction::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	Function::draw(canvas, origin, scale);

	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);

	xt::xarray<double> drawWeights = weights.getParameters();

	const double RANGE = 3.0; // Controls the range of the plot to display (-RANGE, RANGE)
	const int RESOLUTION = 10; // Controls the number of points to estimate
	const float RESCALE = (1.0 / RANGE) * DRAW_LEN * scale;

	ImVec2 position(0, origin.y);
	const double LAYER_WIDTH = NeuralLayer::getLayerWidth(numUnits, scale);
	std::vector<size_t> pointsShape;
	pointsShape.push_back(RESOLUTION);
	pointsShape.push_back(numInputs-1); // Remove bias
	xt::xarray<double> points = xt::zeros<double>(pointsShape);
	for (int i = 0; i < numUnits; i++)
	{
		position.x = NeuralLayer::getNeuronX(origin.x, LAYER_WIDTH, i, scale);
		for (int r = 0; r < RESOLUTION; r++)
		{
			points(r, 0) = RANGE * (2.0 * r) / (RESOLUTION - 1.0) - RANGE;
		}
		auto ys = feedForward(points); // TODO: Thread safety

		for (int r = 0; r < (RESOLUTION - 1); r++)
		{
			ImVec2 l_start(position.x + (points(r, 0) * RESCALE), position.y - (ys(r, i) * RESCALE));
			ImVec2 l_end(position.x + (points(r + 1, 0) * RESCALE), position.y - (ys(r + 1, i) * RESCALE));
			canvas->AddLine(l_start, l_end, BLACK);
		}
	}
}