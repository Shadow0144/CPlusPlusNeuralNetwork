#pragma once

#include <vector>
#include "NeuralLayer.h"
#include "ErrorFunction.h"

class NetworkVisualizer;

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(bool drawingEnabled = true);
	~NeuralNetwork();

	void addInputLayer(std::vector<size_t> inputShape);
	void addDenseLayer(DenseActivationFunction layerFunction, size_t numUnits);
	void addSoftmaxLayer(int axis = -1);
	void addConvolutionLayer(ConvolutionActivationFunction layerFunction, size_t numKernels, std::vector<size_t> convolutionShape, size_t stride = 1);
	void addPoolingLayer(PoolingActivationFunction layerFunction, std::vector<size_t> poolingShape);
	void addFlattenLayer(int numOutputs);

	xt::xarray<double> feedForward(xt::xarray<double> inputs);
	bool backPropagate(xt::xarray<double> inputs, xt::xarray<double> targets); // Single step
	void train(xt::xarray<double> inputs, xt::xarray<double> targets); // Train until a condition is met

	void setTrainingParameters(ErrorFunction* errorFunction, int maxIterations,
		double minError, double errorConvergenceThreshold, double weightConvergenceThreshold);

	void setClassificationVisualizationParameters(int rows, int cols, ImColor* classColors);

	double getError(xt::xarray<double> predicted, xt::xarray<double> actual);

	int getVerbosity();
	void setVerbosity(int verbosity);

	int getBatchSize();
	void setBatchSize(int batchSize);

	int getOutputRate();
	void setOutputRate(int outputRate);
	bool getDrawingEnabled();
	void setDrawingEnabled(bool drawingEnabled);
	void displayRegressionEstimation();
	void displayClassificationEstimation();

	void draw(xt::xarray<double> inputs, xt::xarray<double> targets);
	void draw(ImDrawList* canvas, ImVec2 origin, double scale, xt::xarray<double> target_xs, xt::xarray<double> target_ys);

private:
	int verbosity;
	bool drawingEnabled;
	size_t layerCount;
	vector<size_t> inputShape;
	vector<NeuralLayer*>* layers;
	ErrorFunction* errorFunction;
	int maxIterations;
	double minError;
	double errorConvergenceThreshold;
	double weightConvergenceThreshold;
	int outputRate;
	int batchSize;
	NetworkVisualizer* visualizer;
	ImColor* colors; // TODO

	enum class LearningState // For printing output
	{
		untrained,
		training,
		trained
	};

	void output(LearningState state, int iteration, xt::xarray<double> inputs, xt::xarray<double> targets, xt::xarray<double> predicted);
};