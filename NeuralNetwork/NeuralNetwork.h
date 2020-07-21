#pragma once

#include <vector>
#include "Neuron.h"
#include "ErrorFunction.h"

class NetworkVisualizer;

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(int layerCount, int* layerShapes, ActivationFunction* layerFunctions);
	~NeuralNetwork();

	MatrixXd feedForward(MatrixXd inputs);
	bool backPropagate(MatrixXd inputs, MatrixXd targets); // Single step
	void train(MatrixXd inputs, MatrixXd targets); // Train until a condition is met

	void setTrainingParameters(ErrorFunction* errorFunction, int maxIterations,
		double minError, double errorConvergenceThreshold, double weightConvergenceThreshold);

	double getError(MatrixXd predicted, MatrixXd actual);

	int getVerbosity();
	void setVerbosity(int verbosity);

	int getDrawRate();
	void setDrawRate(int drawRate);
	bool getDrawingEnabled();
	void setDrawingEnabled(bool drawingEnabled);
	void draw(MatrixXd target_xs, MatrixXd target_ys);

private:
	int verbosity;
	bool drawingEnabled;
	int layerCount;
	int* layerShapes;
	vector<Neuron*>* layers;
	ErrorFunction* errorFunction;
	int maxIterations;
	double minError;
	double errorConvergenceThreshold;
	double weightConvergenceThreshold;
	int drawRate;
	NetworkVisualizer* visualizer;

	enum class LearningState // For printing output
	{
		untrained,
		training,
		trained
	};

	void Output(LearningState state, int iteration, MatrixXd inputs, MatrixXd targets, MatrixXd predicted);

	friend class NetworkVisualizer;
};