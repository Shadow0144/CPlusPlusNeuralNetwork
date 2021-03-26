#include "AdamOptimizer.h"

const double INTERNAL_BATCH_LIMIT = 20;

using namespace std;

const std::string AdamOptimizer::ETA = "eta"; // Parameter string [REQUIRED]
const std::string AdamOptimizer::BATCH_SIZE = "batchSize"; // Parameter string [OPTIONAL]
const std::string AdamOptimizer::BETA1 = "beta1"; // Parameter string [OPTIONAL]
const std::string AdamOptimizer::BETA2 = "beta2"; // Parameter string [OPTIONAL]
const std::string AdamOptimizer::EPSILON = "epsilon"; // Parameter string [OPTIONAL]

AdamOptimizer::AdamOptimizer(vector<NeuralLayer*>* layers, double eta, int batchSize, double beta1, double beta2, double epsilon) : Optimizer(layers)
{
	this->eta = eta;
	this->batchSize = batchSize;
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->epsilon = epsilon;
}

AdamOptimizer::AdamOptimizer(vector<NeuralLayer*>* layers, std::map<std::string, double> additionalParameters) : Optimizer(layers)
{
	if (additionalParameters.find(ETA) == additionalParameters.end())
	{
		throw std::invalid_argument(std::string("Missing required parameter: ") +
			"SGDOptimizer::ETA" + " (\"" + ETA + "\")");
	}
	else
	{
		this->eta = additionalParameters[ETA];
		if (additionalParameters.find(BATCH_SIZE) == additionalParameters.end())
		{
			this->batchSize = -1;
		}
		else
		{
			this->batchSize = additionalParameters[BATCH_SIZE];
		}
		if (additionalParameters.find(BATCH_SIZE) == additionalParameters.end())
		{
			this->batchSize = -1;
		}
		else
		{
			this->batchSize = additionalParameters[BATCH_SIZE];
		}
		if (additionalParameters.find(BETA1) == additionalParameters.end())
		{
			this->beta1 = 0.9;
		}
		else
		{
			this->beta1 = additionalParameters[BETA1];
		}
		if (additionalParameters.find(BETA2) == additionalParameters.end())
		{
			this->beta2 = 0.999;
		}
		else
		{
			this->beta2 = additionalParameters[BETA2];
		}
		if (additionalParameters.find(EPSILON) == additionalParameters.end())
		{
			this->epsilon = 1e-8;
		}
		else
		{
			this->epsilon = additionalParameters[EPSILON];
		}
	}
}

double AdamOptimizer::backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
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

double AdamOptimizer::backPropagateBatch(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
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

xt::xarray<double> AdamOptimizer::getDeltaWeight(long parameterID, const xt::xarray<double>& gradient)
{
	if (m.find(parameterID) == m.end() ||
		v.find(parameterID) == v.end())
	{
		m[parameterID] = xt::zeros<double>(gradient.shape());
		v[parameterID] = xt::zeros<double>(gradient.shape());
	}
	else { }
	m[parameterID] = (beta1 * m[parameterID]) + ((1 - beta1) * gradient);
	v[parameterID] = (beta2 * v[parameterID]) + ((1 - beta2) * xt::pow(gradient, 2.0));
	auto mHat = m[parameterID] / (1 - beta1);
	auto vHat = v[parameterID] / (1 - beta2);
	xt::xarray<double> optimizedGradient = -eta / (xt::pow(vHat, 0.5) + epsilon) * mHat;
	return optimizedGradient;
}