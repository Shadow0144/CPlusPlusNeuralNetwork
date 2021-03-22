#include "SGDOptimizer.h"

const double INTERNAL_BATCH_LIMIT = 20;

using namespace std;

SGDOptimizer::SGDOptimizer(vector<NeuralLayer*>* layers) : Optimizer(layers)
{

}

bool SGDOptimizer::backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
{
	bool converged = true;

	if (errorFunction != nullptr)
	{
		// We need to limit the size of the data being processed at once for memory reasons, but the update will still be on the entire batch
		const int N = inputs.shape()[0];
		const int INTERNAL_BATCHES = ceil(N / INTERNAL_BATCH_LIMIT);
		int iBatchSize = N / INTERNAL_BATCHES;

		size_t layerCount = layers->size();
		auto shape = layers->at(layerCount - 1)->getOutputShape();
		shape[0] = N;
		//shape.insert(shape.begin(), N);
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
			xt::xarray<double> errors = errorFunction->getDerivativeOfError(predicted, xt::strided_view(targets, iBatchSV));

			// Backpropagate through the layers until the input layer
			for (int l = (layerCount - 1); l > 0; l--)
			{
				errors = layers->at(l)->backPropagate(errors);
			}
		}

		// Apply the backpropagation
		for (int l = 0; l < layerCount; l++)
		{
			double deltaSum = layers->at(l)->applyBackPropagate();
			converged = (deltaSum < weightConvergenceThreshold) && converged;
		}
	}
	else { }

	return converged;
}