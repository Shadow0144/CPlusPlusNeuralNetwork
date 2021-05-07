#pragma once

#include "NeuralLayer.h"
#include "Optimizer.h"
#include "ActivationFunction.h"

#pragma warning(push, 0)
#include <vector>
#include <map>
#pragma warning(pop)

enum class OptimizerType
{
	SGD, // Stochastic Gradient Descent
	Adagrad, // Adagrad
	Adadelta, // Adadelta
	Adamax, // Adamax
	RProp, // Resilient Propagation
	RMSProp, // Root Mean Square Propagation
	Adam, // Adaptive Moment Estimation
	Nadam, // Nesterov-accelerated Adaptive Moment Estimation
	AMSGrad, // AMSGrad
	Ftrl // Follow The Regularized Leader (-Proximal)
};

enum class LossFunctionType
{
	CrossEntropy,
	MeanSquaredError
};

enum class StoppingCondition
{
	Max_Epochs = 0, // Maximum number of epochs
	Min_Loss = 1, // Minimum loss threshold
	Min_Delta_Loss = 2, // Minimum change in loss between epochs
	Min_Delta_Params = 3 // Minimum change in parameters between epochs
};

class LossFunction;

class NetworkVisualizer;

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(bool drawingEnabled = true);
	~NeuralNetwork();

	void addInputLayer(const std::vector<size_t>& inputShape);
	void addDenseLayer(ActivationFunctionType layerFunction, size_t numUnits, 
		std::map<string, double> additionalParameters = std::map<string, double>(), bool addBias = true);
	void addActivationFunctionLayer(ActivationFunctionType layerFunction, std::map<string, double> additionalParameters = std::map<string, double>());
	void addMaxoutLayer(size_t numUnits, size_t numFunctions, bool addBias = true);
	void addSoftmaxLayer(int axis = -1);
	void addConvolution1DLayer(size_t numKernels, const std::vector<size_t>& convolutionShape, const std::vector<size_t>& stride = { 1 }, bool addBias = false, ActivationFunctionType activationFunctionType = ActivationFunctionType::Identity, std::map<std::string, double> additionalParameters = std::map<std::string, double>());
	void addConvolution2DLayer(size_t numKernels, const std::vector<size_t>& convolutionShape, const std::vector<size_t>& stride = { 1 }, bool addBias = false, ActivationFunctionType activationFunctionType = ActivationFunctionType::Identity, std::map<std::string, double> additionalParameters = std::map<std::string, double>());
	void addConvolution3DLayer(size_t numKernels, const std::vector<size_t>& convolutionShape, const std::vector<size_t>& stride = { 1 }, bool addBias = false, ActivationFunctionType activationFunctionType = ActivationFunctionType::Identity, std::map<std::string, double> additionalParameters = std::map<std::string, double>());
	void addAveragePooling1DLayer(const std::vector<size_t>& poolingShape, bool hasChannels = true);
	void addAveragePooling2DLayer(const std::vector<size_t>& poolingShape, bool hasChannels = true);
	void addAveragePooling3DLayer(const std::vector<size_t>& poolingShape, bool hasChannels = true);
	void addMaxPooling1DLayer(const std::vector<size_t>& poolingShape, bool hasChannels = true);
	void addMaxPooling2DLayer(const std::vector<size_t>& poolingShape, bool hasChannels = true);
	void addMaxPooling3DLayer(const std::vector<size_t>& poolingShape, bool hasChannels = true);
	void addFlattenLayer();
	void addSqueezeLayer(const std::vector<size_t>& squeezeDims = std::vector<size_t>());
	void addReshapeLayer(const std::vector<size_t>& newShape);
	void addDropoutLayer(double dropRate = 0.5);

	xt::xarray<double> predict(const xt::xarray<double>& inputs) const; // Does not update training values
	void train(const xt::xarray<double>& inputs, const xt::xarray<double>& targets, int maxEpochs = -1); // Train until a condition is met

	void setOptimizer(OptimizerType optimizerType, std::map<std::string, double> additionalParameters = std::map<std::string, double>());

	void setLossFunction(LossFunctionType lossFunctionType, double lambda1 = 0.0, double lambda2 = 0.0);
	void enableStoppingCondition(StoppingCondition condition, double threshold);
	void disableStoppingCondition(StoppingCondition condition);
	bool getStoppingConditionEnabled(StoppingCondition condition);
	double getStoppingConditionThreshold(StoppingCondition condition);

	//void setClassificationVisualizationParameters(int rows, int cols, ImColor* classColors);

	double getLoss(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);
	double getRegularizationLoss(double lambda1, double lambda2) const;

	int getVerbosity();
	void setVerbosity(int verbosity);

	int getOutputRate();
	void setOutputRate(int outputRate);

	void resetEpoch();

	void loadParameters(string folderName);
	void saveParameters(string folderName);
	void enableAutosave(string folderName, int perIterations);
	void disableAutosave();

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
	LossFunction* lossFunction;
	Optimizer* optimizer;
	int maxEpochs;
	double minLoss;
	double lossConvergenceThreshold;
	double weightConvergenceThreshold;
	int outputRate;
	NetworkVisualizer* visualizer;
	bool* stoppingConditionFlags;

	int currentEpoch;

	enum class LearningState // For printing output
	{
		untrained,
		training,
		trained
	};

	int autosaveFrequency;
	string autosaveFileName;
	bool autosaveEnabled;

	void setupDrawing(const xt::xarray<double>& inputs, const xt::xarray<double>& targets);
	void updateDrawing(const xt::xarray<double>& predicted);
	void output(LearningState state, int epoch, const xt::xarray<double>& inputs, 
						const xt::xarray<double>& targets, const xt::xarray<double>& predicted);
};