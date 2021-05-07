#include "NeuralNetwork.h"

#pragma warning(push, 0)
#ifdef __GNUC__
#define LINUX
#else
#define WINDOWS
#endif
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#pragma warning(pop)

#include "NetworkVisualizer.h"

#include "InputNeuralLayer.h"
#include "DenseNeuralLayer.h"
#include "ActivationFunctionNeuralLayer.h"
#include "SoftmaxNeuralLayer.h"
#include "MaxoutNeuralLayer.h"
#include "Convolution1DNeuralLayer.h"
#include "Convolution2DNeuralLayer.h"
#include "Convolution3DNeuralLayer.h"
#include "AveragePooling1DNeuralLayer.h"
#include "AveragePooling2DNeuralLayer.h"
#include "AveragePooling3DNeuralLayer.h"
#include "MaxPooling1DNeuralLayer.h"
#include "MaxPooling2DNeuralLayer.h"
#include "MaxPooling3DNeuralLayer.h"
#include "FlattenNeuralLayer.h"
#include "SqueezeNeuralLayer.h"
#include "ReshapeNeuralLayer.h"
#include "DropoutNeuralLayer.h"

#include "SGDOptimizer.h"
#include "AdagradOptimizer.h"
#include "AdadeltaOptimizer.h"
#include "AdamaxOptimizer.h"
#include "RPropOptimizer.h"
#include "RMSPropOptimizer.h"
#include "AdamOptimizer.h"
#include "NadamOptimizer.h"
#include "AMSGradOptimizer.h"
#include "FtrlOptimizer.h"

#include "LossFunction.h"
#include "CrossEntropyLossFunction.h"
#include "MeanSquareErrorLossFunction.h"

#include "NeuralNetworkFileHelper.h"

#include "NetworkExceptions.h"

#include "Test.h"

#pragma warning(push, 0)
#include <thread>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <xtensor/xview.hpp>
#include <filesystem>
#pragma warning(pop)

const double INTERNAL_BATCH_LIMIT = 20;

NeuralNetwork::NeuralNetwork(bool drawingEnabled)
{
	this->verbosity = 1;
	this->outputRate = 100;
	this->drawingEnabled = drawingEnabled;

	this->autosaveEnabled = false;
	this->autosaveFrequency = 0;

	this->currentEpoch = 0;

	this->optimizer = nullptr;
	this->lossFunction = nullptr;
	this->maxEpochs = -1;
	this->minLoss = -1;
	this->lossConvergenceThreshold = -1;
	this->weightConvergenceThreshold = -1;
	this->stoppingConditionFlags = new bool[4]; // There are four stopping conditions
	for (int i = 0; i < 4; i++)
	{
		stoppingConditionFlags[i] = false;
	}

	if (drawingEnabled)
	{
		visualizer = new NetworkVisualizer(this);
	}
	else 
	{
		visualizer = nullptr;
	}

	this->layerCount = 0;
	this->layers = new vector<NeuralLayer*>();
}

NeuralNetwork::~NeuralNetwork()
{
	if (visualizer != nullptr)
	{
		delete visualizer;
	}
	else { }
	if (layers != nullptr)
	{
		for (int i = 0; i < layerCount; i++) // Loop through the layers
		{
			delete layers->at(i);
		}
		delete layers;
	}
	else { }
	if (lossFunction != nullptr)
	{
		delete lossFunction;
	}
	else { }
	if (optimizer != nullptr)
	{
		delete optimizer;
	}
	else { }
}

void NeuralNetwork::addInputLayer(const std::vector<size_t>& inputShape)
{
	if (layerCount != 0) // Ensure this is the first layer
	{
		throw InputLayerConfigurationException();
	}
	else { }
	this->inputShape = inputShape; 
	InputNeuralLayer* layer = new InputNeuralLayer(inputShape);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addDenseLayer(ActivationFunctionType layerFunction, size_t numUnits, std::map<string, double> additionalParameters, bool addBias)
{
	DenseNeuralLayer* layer = new DenseNeuralLayer(layerFunction, layers->at(layerCount - 1), numUnits, additionalParameters, addBias);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addActivationFunctionLayer(ActivationFunctionType layerFunction, std::map<string, double> additionalParameters)
{
	ActivationFunctionNeuralLayer* layer = new ActivationFunctionNeuralLayer(layerFunction, layers->at(layerCount - 1), additionalParameters);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addMaxoutLayer(size_t numUnits, size_t numFuctions, bool addBias)
{
	MaxoutNeuralLayer* layer = new MaxoutNeuralLayer(layers->at(layerCount - 1), numUnits, numFuctions, addBias);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addSoftmaxLayer(int axis)
{
	SoftmaxNeuralLayer* layer = new SoftmaxNeuralLayer(layers->at(layerCount - 1), axis);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addConvolution1DLayer(size_t numKernels, const std::vector<size_t>& convolutionShape, 
											const std::vector<size_t>& stride, bool padded, bool addBias,
											ActivationFunctionType activationFunctionType, 
											std::map<string, double> additionalParameters)
{
	Convolution1DNeuralLayer* layer = new Convolution1DNeuralLayer(layers->at(layerCount - 1), numKernels, convolutionShape, 
																	stride, padded, addBias, activationFunctionType,
																	additionalParameters);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addConvolution2DLayer(size_t numKernels, const std::vector<size_t>& convolutionShape,
											const std::vector<size_t>& stride, bool padded, bool addBias,
											ActivationFunctionType activationFunctionType,
											std::map<string, double> additionalParameters)
{
	Convolution2DNeuralLayer* layer = new Convolution2DNeuralLayer(layers->at(layerCount - 1), numKernels, convolutionShape,
																	stride, padded, addBias, activationFunctionType,
																	additionalParameters);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addConvolution3DLayer(size_t numKernels, const std::vector<size_t>& convolutionShape,
											const std::vector<size_t>& stride, bool padded, bool addBias,
											ActivationFunctionType activationFunctionType,
											std::map<string, double> additionalParameters)
{
	Convolution3DNeuralLayer* layer = new Convolution3DNeuralLayer(layers->at(layerCount - 1), numKernels, convolutionShape,
																	stride, padded, addBias, activationFunctionType,
																	additionalParameters);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addAveragePooling1DLayer(const std::vector<size_t>& poolingShape, bool hasChannels)
{
	AveragePooling1DNeuralLayer* layer = new AveragePooling1DNeuralLayer(layers->at(layerCount - 1), poolingShape, hasChannels);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addAveragePooling2DLayer(const std::vector<size_t>& poolingShape, bool hasChannels)
{
	AveragePooling2DNeuralLayer* layer = new AveragePooling2DNeuralLayer(layers->at(layerCount - 1), poolingShape, hasChannels);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addAveragePooling3DLayer(const std::vector<size_t>& poolingShape, bool hasChannels)
{
	AveragePooling3DNeuralLayer* layer = new AveragePooling3DNeuralLayer(layers->at(layerCount - 1), poolingShape, hasChannels);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addMaxPooling1DLayer(const std::vector<size_t>& poolingShape, bool hasChannels)
{
	MaxPooling1DNeuralLayer* layer = new MaxPooling1DNeuralLayer(layers->at(layerCount - 1), poolingShape, hasChannels);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addMaxPooling2DLayer(const std::vector<size_t>& poolingShape, bool hasChannels)
{
	MaxPooling2DNeuralLayer* layer = new MaxPooling2DNeuralLayer(layers->at(layerCount - 1), poolingShape, hasChannels);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addMaxPooling3DLayer(const std::vector<size_t>& poolingShape, bool hasChannels)
{
	MaxPooling3DNeuralLayer* layer = new MaxPooling3DNeuralLayer(layers->at(layerCount - 1), poolingShape, hasChannels);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addFlattenLayer()
{
	FlattenNeuralLayer* layer = new FlattenNeuralLayer(layers->at(layerCount - 1));
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addSqueezeLayer(const std::vector<size_t>& squeezeDims)
{
	SqueezeNeuralLayer* layer = new SqueezeNeuralLayer(layers->at(layerCount - 1), squeezeDims);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addReshapeLayer(const std::vector<size_t>& newShape)
{
	ReshapeNeuralLayer* layer = new ReshapeNeuralLayer(layers->at(layerCount - 1), newShape);
	layers->push_back(layer);
	layerCount++;
}

void NeuralNetwork::addDropoutLayer(double dropRate)
{
	DropoutNeuralLayer* layer = new DropoutNeuralLayer(layers->at(layerCount - 1), dropRate);
	layers->push_back(layer);
	layerCount++;
}

xt::xarray<double> NeuralNetwork::predict(const xt::xarray<double>& inputs) const
{
	const int N = inputs.shape()[0];
	const int INTERNAL_BATCHES = ceil(N / INTERNAL_BATCH_LIMIT);
	int iBatchSize = N / INTERNAL_BATCHES;

	auto shape = layers->at(layerCount - 1)->getOutputShape();
	//shape[0] = N;
	shape.insert(shape.begin(), N);
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
		xt::xarray<double> predictedBatch = xt::strided_view(inputs, iBatchSV);

		for (int i = 0; i < layerCount; i++) // Loop through the layers
		{
			predictedBatch = layers->at(i)->feedForward(predictedBatch);
		}

		xt::strided_view(predicted, iBatchSV) = predictedBatch;
	}

	return predicted;
}

void NeuralNetwork::train(const xt::xarray<double>& inputs, const xt::xarray<double>& targets, int maxEpochs)
{
	setupDrawing(inputs, targets);

	if (maxEpochs > -1)
	{
		enableStoppingCondition(StoppingCondition::Max_Epochs, maxEpochs);
	}
	else
	{
		disableStoppingCondition(StoppingCondition::Max_Epochs);
	}

	if (optimizer != nullptr)
	{
		if (lossFunction != nullptr)
		{
			xt::xarray<double> predicted = predict(inputs);
			double loss = abs(getLoss(predicted, targets));
			output(LearningState::untrained, currentEpoch, inputs, targets, predicted);

			updateDrawing(predicted);

			cout << "Beginning training" << endl << endl;

			// Check if there is a special optimized synergy between the gradients of the last layer and the loss function
			lossFunction->checkForOptimizedGradient(layers->at(layers->size() - 1));

			//currentEpoch = 0; //int t = 0; // This is set either in the constructor or when loading
			bool converged = false;
			double lastLoss = loss;
			double deltaLoss = loss;
			while
				(!((stoppingConditionFlags[((int)(StoppingCondition::Max_Epochs))]) && (currentEpoch >= maxEpochs)) &&
					!((stoppingConditionFlags[((int)(StoppingCondition::Min_Loss))]) && (loss < 0 || loss > minLoss)) &&
					!((stoppingConditionFlags[((int)(StoppingCondition::Min_Delta_Loss))]) && (deltaLoss < lossConvergenceThreshold)) &&
					!((stoppingConditionFlags[((int)(StoppingCondition::Min_Delta_Params))]) && (!converged)))
			{
				double deltaSum = optimizer->backPropagate(inputs, targets);
				converged = (deltaSum < weightConvergenceThreshold);

				predicted = predict(inputs);
				loss = getLoss(predicted, targets);
				deltaLoss = abs(lastLoss - loss);
				lastLoss = loss;

				updateDrawing(predicted);

				currentEpoch++;

				if (currentEpoch % outputRate == 0 && currentEpoch != 0)
				{
					output(LearningState::training, currentEpoch, inputs, targets, predicted);
				}
				else { }

				if (autosaveEnabled && currentEpoch % autosaveFrequency == 0)
				{
					saveParameters(autosaveFileName);
					cout << "Epoch " << currentEpoch << " parameters saved" << endl << endl;
				}
				else { }
			}

			if (verbosity >= 1)
			{
				if ((stoppingConditionFlags[((int)(StoppingCondition::Min_Delta_Params))]) && (converged))
				{
					cout << "Weights have converged" << endl << endl;
				}
				else if ((stoppingConditionFlags[((int)(StoppingCondition::Min_Delta_Loss))]) && (deltaLoss <= lossConvergenceThreshold))
				{
					cout << "Loss has converged" << endl << endl;
				}
				else if ((stoppingConditionFlags[((int)(StoppingCondition::Min_Loss))]) && (loss >= 0 && loss <= minLoss))
				{
					cout << "Minimum loss condition reached" << endl << endl;
				}
				else if ((stoppingConditionFlags[((int)(StoppingCondition::Max_Epochs))]) && (currentEpoch == maxEpochs))
				{
					cout << "Maximum epochs reached" << endl << endl;
				}
				else { }
			}
			else { }

			output(LearningState::trained, currentEpoch, inputs, targets, predicted);

			cout << "Training complete" << endl << endl;

			updateDrawing(predicted);

			system("pause");
		}
		else
		{
			cout << "No loss function set" << endl;
		}
	}
	else
	{
		cout << "No optimizer set" << endl;
	}
}

void NeuralNetwork::output(LearningState state, int epoch, const xt::xarray<double>& inputs, const xt::xarray<double>& targets, const xt::xarray<double>& predicted)
{
	if (verbosity >= 1)
	{
		string stateString = "";
		switch (state)
		{
			case LearningState::untrained:
				stateString = "Initial";
				break;
			case LearningState::training:
				stateString = "Training";
				break;
			case LearningState::trained:
				stateString = "Trained";
				break;
		}
		cout << stateString << ": " << endl;
		if (verbosity >= 2)
		{
			const int N = inputs.shape()[0];
			for (int i = 0; i < N; i++)
			{
				cout << "Feedforward " << stateString << ": X: " << inputs(i) << " Y': " << targets(i) << " Y: " << predicted(i) << endl;
			}
		}
		else { }
		cout << "Epochs: " << epoch << endl;
		cout << "Loss: " << getLoss(predicted, targets) << endl;
		cout << endl;
	}
	else { }
}

void NeuralNetwork::setOptimizer(OptimizerType optimizerType, std::map<string, double> additionalParameters)
{
	if (optimizer != nullptr)
	{
		delete optimizer;
	}
	else { }

	switch (optimizerType)
	{
		case OptimizerType::SGD:
			this->optimizer = new SGDOptimizer(layers, additionalParameters);
			break;
		case OptimizerType::Adagrad:
			this->optimizer = new AdagradOptimizer(layers, additionalParameters);
			break;
		case OptimizerType::Adadelta:
			this->optimizer = new AdadeltaOptimizer(layers, additionalParameters);
			break;
		case OptimizerType::Adamax:
				this->optimizer = new AdamaxOptimizer(layers, additionalParameters);
				break;
		case OptimizerType::RProp:
			this->optimizer = new RPropOptimizer(layers, additionalParameters);
			break;
		case OptimizerType::RMSProp:
			this->optimizer = new RMSPropOptimizer(layers, additionalParameters);
			break;
		case OptimizerType::Adam:
			this->optimizer = new AdamOptimizer(layers, additionalParameters);
			break;
		case OptimizerType::Nadam:
			this->optimizer = new NadamOptimizer(layers, additionalParameters);
			break;
		case OptimizerType::AMSGrad:
			this->optimizer = new AMSGradOptimizer(layers, additionalParameters);
			break;
		case OptimizerType::Ftrl:
			this->optimizer = new FtrlOptimizer(layers, additionalParameters);
			break;
	}

	if (lossFunction != nullptr)
	{
		optimizer->setLossFunction(lossFunction);
	}
	else { }
}

void NeuralNetwork::setLossFunction(LossFunctionType errorFunctionType, double lambda1, double lambda2)
{
	if (lossFunction != nullptr)
	{
		delete lossFunction;
	}
	else { }

	switch (errorFunctionType)
	{
		case LossFunctionType::CrossEntropy:
			this->lossFunction = new CrossEntropyLossFunction();
			break;
		case LossFunctionType::MeanSquaredError:
			this->lossFunction = new MeanSquareErrorLossFunction();
			break;
	}
	this->lossFunction->setL1RegularizationStrength(lambda1);
	this->lossFunction->setL2RegularizationStrength(lambda2);

	if (optimizer != nullptr)
	{
		optimizer->setLossFunction(lossFunction);
	}
	else { }
}

void NeuralNetwork::enableStoppingCondition(StoppingCondition condition, double threshold)
{
	stoppingConditionFlags[((int)(condition))] = true;
	switch (condition)
	{
		case StoppingCondition::Max_Epochs:
			this->maxEpochs = threshold;
			break;
		case StoppingCondition::Min_Loss:
			this->minLoss = threshold;
			break;
		case StoppingCondition::Min_Delta_Loss:
			this->lossConvergenceThreshold = threshold;
			break;
		case StoppingCondition::Min_Delta_Params:
			this->weightConvergenceThreshold = threshold;
			break;
	}
}

void NeuralNetwork::disableStoppingCondition(StoppingCondition condition)
{
	stoppingConditionFlags[((int)(condition))] = false;
}

bool NeuralNetwork::getStoppingConditionEnabled(StoppingCondition condition)
{
	return stoppingConditionFlags[((int)(condition))];
}

double NeuralNetwork::getStoppingConditionThreshold(StoppingCondition condition)
{
	double rValue = 0;
	switch (condition)
	{
		case StoppingCondition::Max_Epochs:
			rValue = this->maxEpochs;
			break;
		case StoppingCondition::Min_Loss:
			rValue = this->minLoss;
			break;
		case StoppingCondition::Min_Delta_Loss:
			rValue = this->lossConvergenceThreshold;
			break;
		case StoppingCondition::Min_Delta_Params:
			rValue = this->weightConvergenceThreshold;
			break;
	}
	return rValue;
}

double NeuralNetwork::getLoss(const xt::xarray<double>& predicted, const xt::xarray<double>& actual)
{
	return lossFunction->getLoss(this, predicted, actual);
}

double NeuralNetwork::getRegularizationLoss(double lambda1, double lambda2) const
{
	double loss = 0.0;
	if (lambda1 != 0.0 || lambda2 != 0.0)
	{
		for (int i = 0; i < layerCount; i++) // Loop through the layers
		{
			loss += layers->at(i)->getRegularizationLoss(lambda1, lambda2);
		}
	}
	else { }
	return loss;
}

int NeuralNetwork::getVerbosity()
{
	return verbosity;
}

void NeuralNetwork::setVerbosity(int verbosity)
{
	this->verbosity = verbosity;
}

int NeuralNetwork::getOutputRate()
{
	return outputRate;
}

void NeuralNetwork::setOutputRate(int outputRate)
{
	this->outputRate = outputRate;
}

void NeuralNetwork::resetEpoch()
{
	currentEpoch = 0;
}

std::string getCurrentDir()
{
	char buffer[FILENAME_MAX];
	#pragma warning(suppress : 6031)
	GetCurrentDir(buffer, FILENAME_MAX);
	string currentWorkingDir(buffer);
	return currentWorkingDir;
}

void NeuralNetwork::loadParameters(string folderName)
{
	string fullFolderName = getCurrentDir().append("\\" + folderName);
	string fileName = fullFolderName + "\\" + folderName;
	if (NeuralNetworkFileHelper::folderExists(fullFolderName) && NeuralNetworkFileHelper::fileExists(fileName + ".nnp"))
	{
		ifstream loadFile;
		loadFile.open(fileName + ".nnp");
		loadFile >> currentEpoch;
		loadFile.close();
		for (int l = 0; l < layerCount; l++)
		{
			layers->at(l)->loadParameters(fileName + "_" + to_string(l)); // The individual layers add their own file endings as needed
		}
		cout << "Loading complete" << endl << endl;
	}
	else
	{
		cout << "No parameters to load" << endl << endl;
	}
}

void NeuralNetwork::saveParameters(string folderName)
{	
	bool success = true;
	std::filesystem::path folderPath = getCurrentDir().append("\\" + folderName);
	bool exists = std::filesystem::exists(folderPath);
	if (!exists)
	{
		success = std::filesystem::create_directory(folderPath);
	}
	else { }
	if (success)
	{
		string fileName = folderPath.string().append("\\" + folderName);
		ofstream saveFile;
		saveFile.open(fileName + ".nnp");
		saveFile << currentEpoch;
		saveFile.close();
		for (int l = 0; l < layerCount; l++)
		{
			layers->at(l)->saveParameters(fileName + "_" + to_string(l)); // The individual layers add their own file endings as needed
		}
	}
	else
	{
		cout << "Cannot create or access directory" << endl << endl;
	}
}

void NeuralNetwork::enableAutosave(string folderName, int perIterations)
{
	this->autosaveFileName = folderName;
	this->autosaveFrequency = perIterations;
	this->autosaveEnabled = true;
}

void NeuralNetwork::disableAutosave()
{
	this->autosaveEnabled = false;
}

bool NeuralNetwork::getDrawingEnabled()
{
	return drawingEnabled;
}

void NeuralNetwork::setDrawingEnabled(bool drawingEnabled)
{
	this->drawingEnabled = drawingEnabled;
	if (drawingEnabled && visualizer == nullptr)
	{
		visualizer = new NetworkVisualizer(this);
	}
	else if (!drawingEnabled && visualizer != nullptr)
	{ 
		delete visualizer;
		visualizer = nullptr;
	}
	else { }
}

void NeuralNetwork::displayRegressionEstimation()
{
	if (drawingEnabled)
	{
		visualizer->addFunctionVisualization();
	}
	else { }
}

void NeuralNetwork::displayClassificationEstimation(int rows, int cols, ImColor* colors)
{
	if (drawingEnabled)
	{
		visualizer->addClassificationVisualization(rows, cols, colors);
	}
	else { }
}

void NeuralNetwork::setupDrawing(const xt::xarray<double>& inputs, const xt::xarray<double>& targets)
{
	if (drawingEnabled)
	{
		visualizer->setTargets(inputs, targets);
	}
	else { }
}

void NeuralNetwork::updateDrawing(const xt::xarray<double>& predicted)
{
	if (drawingEnabled)
	{
		visualizer->setPredicted(predicted);
	}
	else { }
}

void NeuralNetwork::draw(ImDrawList* canvas, ImVec2 origin, double scale, const xt::xarray<double>& target_xs, const xt::xarray<double>& target_ys)
{
	// Calculate the drawing space parameters
	const double Y_SHIFT = scale * 120.0;
	const double X_SHIFT = scale * 120.0;
	
	// Draw the network
	// Find the vertical start point
	double x = origin.x;
	double y = origin.y - (layerCount * Y_SHIFT / 2.0) + (Y_SHIFT / 2.0);
	for (int l = 0; l < layerCount; l++)
	{
		// Set the offset and draw the layer
		ImVec2 offset = ImVec2(x, y);
		layers->at(l)->draw(canvas, offset, scale, (l == (layerCount-1)));
		y += Y_SHIFT;
	}
}