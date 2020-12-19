#pragma once

#include "NeuralLayer.h"
#include "ErrorFunction.h"

#pragma warning(push, 0)
#include <vector>
#pragma warning(pop)

enum class ActivationFunctionType
{
	None,
	ReLU,
	AbsoluteReLU,
	//CReLU,
	ELU,
	SELU,
	GELU,
	LeakyReLU,
	//PReLU,
	ReLU6,
	ReLUn,
	Softplus,
	Exponential,
	Quadratic,
	Sigmoid,
	Tanh,
	HardSigmoid,
	Softsign,
	//Swish,
	//Maxout
};

class NetworkVisualizer;

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(bool drawingEnabled = true);
	~NeuralNetwork();

	void addInputLayer(const std::vector<size_t>& inputShape);
	void addDenseLayer(ActivationFunctionType layerFunction, size_t numUnits);
	void addSoftmaxLayer(int axis = -1);
	void addConvolution1DLayer(size_t numKernels, const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride = 1);
	void addConvolution2DLayer(size_t numKernels, const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride = 1);
	void addConvolution3DLayer(size_t numKernels, const std::vector<size_t>& convolutionShape, size_t inputChannels, size_t stride = 1);
	void addAveragePooling1DLayer(const std::vector<size_t>& poolingShape);
	void addAveragePooling2DLayer(const std::vector<size_t>& poolingShape);
	void addAveragePooling3DLayer(const std::vector<size_t>& poolingShape);
	void addMaxPooling1DLayer(const std::vector<size_t>& poolingShape);
	void addMaxPooling2DLayer(const std::vector<size_t>& poolingShape);
	void addMaxPooling3DLayer(const std::vector<size_t>& poolingShape);
	void addFlattenLayer(int numOutputs);

	xt::xarray<double> feedForward(const xt::xarray<double>& inputs); // Does not update internal values
	xt::xarray<double> feedForwardTrain(const xt::xarray<double>& inputs); // Updates internal values such as last input and last output
	bool backPropagate(const xt::xarray<double>& inputs, const xt::xarray<double>& targets); // Single step
	void train(const xt::xarray<double>& inputs, const xt::xarray<double>& targets); // Train until a condition is met

	void setTrainingParameters(ErrorFunction* errorFunction, int maxIterations,
		double minError, double errorConvergenceThreshold, double weightConvergenceThreshold);

	void setClassificationVisualizationParameters(int rows, int cols, ImColor* classColors);

	double getError(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);

	int getVerbosity();
	void setVerbosity(int verbosity);

	int getBatchSize();
	void setBatchSize(int batchSize);

	int getOutputRate();
	void setOutputRate(int outputRate);
	bool getDrawingEnabled();
	void setDrawingEnabled(bool drawingEnabled);
	void displayRegressionEstimation();
	void displayClassificationEstimation(int rows, int cols, ImColor* colors);

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, const xt::xarray<double>& target_xs, const xt::xarray<double>& target_ys);

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

	enum class LearningState // For printing output
	{
		untrained,
		training,
		trained
	};

	void setupDrawing(const xt::xarray<double>& inputs, const xt::xarray<double>& targets);
	void updateDrawing(const xt::xarray<double>& predicted);
	void output(LearningState state, int iteration, const xt::xarray<double>& inputs, const xt::xarray<double>& targets, const xt::xarray<double>& predicted);
};