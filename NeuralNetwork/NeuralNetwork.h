#pragma once

#include "NeuralLayer.h"
#include "ErrorFunction.h"

#pragma warning(push, 0)
#include <vector>
#include <map>
#pragma warning(pop)

enum class ActivationFunctionType
{
	Identity,
	ReLU,
	AbsoluteReLU,
	CReLU,
	ELU,
	SELU,
	GELU,
	LeakyReLU,
	PReLU,
	ReLU6,
	ReLUn,
	Softplus,
	Exponential,
	Quadratic,
	Sigmoid,
	Tanh,
	HardSigmoid,
	Softsign,
	Swish,
	Maxout
};

enum class ErrorFunctionType
{
	CrossEntropy,
	MeanSquaredError
};

enum class StoppingCondition
{
	Max_Epochs = 0, // Maximum number of epochs
	Min_Error = 1, // Minimum error threshold
	Min_Delta_Error = 2, // Minimum change in error between epochs
	Min_Delta_Params = 3 // Minimum change in parameters between epochs
};

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
	void addMaxoutLayer(size_t numUnits, size_t numFunctions, bool addBias = true);
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
	void train(const xt::xarray<double>& inputs, const xt::xarray<double>& targets, int maxEpochs = -1); // Train until a condition is met

	void setErrorFunction(ErrorFunctionType errorFunctionType);
	void enableStoppingCondition(StoppingCondition condition, double threshold);
	void disableStoppingCondition(StoppingCondition condition);
	bool getStoppingConditionEnabled(StoppingCondition condition);
	double getStoppingConditionThreshold(StoppingCondition condition);

	//void setClassificationVisualizationParameters(int rows, int cols, ImColor* classColors);

	double getError(const xt::xarray<double>& predicted, const xt::xarray<double>& actual);

	int getVerbosity();
	void setVerbosity(int verbosity);

	int getBatchSize();
	void setBatchSize(int batchSize);

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
	ErrorFunction* errorFunction;
	int maxEpochs;
	double minError;
	double errorConvergenceThreshold;
	double weightConvergenceThreshold;
	int outputRate;
	int batchSize;
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

	inline bool folderExists(const std::string& name)
	{
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0 && buffer.st_mode & S_IFDIR);
	};

	inline bool fileExists(const std::string& name)
	{
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	};
};