#include "SGDOptimizer.h"

const double INTERNAL_BATCH_LIMIT = 20;

using namespace std;

const std::string SGDOptimizer::ALPHA = "alpha"; // Parameter string [REQUIRED]
const std::string SGDOptimizer::BATCH_SIZE = "batchSize"; // Parameter string [OPTIONAL]
const std::string SGDOptimizer::MOMENTUM = "momentum"; // Parameter string [OPTIONAL]

SGDOptimizer::SGDOptimizer(vector<NeuralLayer*>* layers, double alpha, int batchSize, double momentum) : Optimizer(layers)
{
	this->alpha = alpha;
	this->batchSize = batchSize;
	this->momentum = momentum;
}

double SGDOptimizer::backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
{
	double deltaWeights = 0.0;

	if (errorFunction != nullptr)
	{
		const int N = inputs.shape()[0];
		int batches = 1;
		if (batchSize > 0) // If there's a batch size set, then use batches
		{
			batches = N / batchSize;
		}
		else { }

		for (int i = 0; i < batches; i++)
		{
			// Set up the batch
			int batchStart = ((i + 0) * batchSize) % N;
			int batchEnd = ((i + 1) * batchSize) % N;
			if ((batchEnd - batchStart) != batchSize)
			{
				batchEnd = N;
			}
			else { }

			xt::xstrided_slice_vector batchSV({ xt::range(batchStart, batchEnd), xt::ellipsis() });
			xt::xarray<double> examples = xt::strided_view(inputs, batchSV);
			xt::xarray<double> exampleTargets = xt::strided_view(targets, batchSV);
			deltaWeights += backPropagateBatch(examples, exampleTargets);
		}
	}
	else { }

	return deltaWeights;
}

double SGDOptimizer::backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
{
	bool converged = true;

	// We need to limit the size of the data being processed at once for memory reasons, but the update will still be on the entire batch
	const int N = inputs.shape()[0];
	const int INTERNAL_BATCHES = ceil(N / INTERNAL_BATCH_LIMIT);
	int iBatchSize = N / INTERNAL_BATCHES;

	size_t layerCount = layers->size();
	auto shape = layers->at(layerCount - 1)->getOutputShape();
	shape[0] = N;
	xt::xarray<double> predicted = xt::xarray<double>(shape);

	for (int i = 0; i < INTERNAL_BATCHES; i++)
	{
		// Set up the batch
		int iBatchStart = ((i + 0) * iBatchSize) % N;
		int iBatchEnd = ((i + 1) * iBatchSize) % N;
		if ((iBatchEnd - iBatchStart) != iBatchSize)
		{
			iBatchEnd = N;
		}
		else { }

		xt::xstrided_slice_vector iBatchSV({ xt::range(iBatchStart, iBatchEnd), xt::ellipsis() });

		// Feed forward and calculate the gradient
		xt::xarray<double> predicted = feedForwardTrain(xt::strided_view(inputs, iBatchSV));
		xt::xarray<double> sigma = errorFunction->getGradient(predicted, xt::strided_view(targets, iBatchSV));

		// Backpropagate through the layers until the input layer
		for (int l = (layerCount - 1); l > 0; l--)
		{
			sigma = layers->at(l)->getGradient(sigma, this);
		}
	}

	// Apply the backpropagation
	double deltaSum = 0.0;
	for (int l = 0; l < layerCount; l++)
	{
		deltaSum += layers->at(l)->applyBackPropagate();
	}

	return deltaSum;
}

xt::xarray<double> SGDOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	xt::xarray<double> optimizedGradient;
	if (momentum > 0.0)
	{
		if (previousVelocity.find(parameterID) == previousVelocity.end())
		{
			previousVelocity[parameterID] = xt::zeros<double>(gradient.shape());
		}
		xt::xarray<double> velocity = momentum * previousVelocity[parameterID] + alpha * gradient;
		optimizedGradient = -velocity;
		previousVelocity[parameterID] = velocity;
	}
	else // Skip storing and updating the velocity if the momentum is 0
	{
		optimizedGradient = -alpha * gradient; // Multiply by the learning rate
	}
	return optimizedGradient;
}