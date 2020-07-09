#pragma once

#include <vector>
#include "Neuron.h"
#include "ErrorFunction.h"

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(int layerCount, int* layerShapes, ActivationFunction* layerFunctions);
	~NeuralNetwork();

	Mat feedForward(Mat inputs);
	bool backPropagate(Mat inputs, Mat targets); // Single step
	void train(Mat inputs, Mat targets); // Train until a condition is met

	void setTrainingParameters(ErrorFunction* errorFunction, int maxIterations,
		float minError, float errorConvergenceThreshold, float weightConvergenceThreshold);

	float getError(Mat predicted, Mat actual);

	int getVerbosity();
	void setVerbosity(int verbosity);

	int getDrawRate();
	void setDrawRate(int drawRate);
	bool getDrawingEnabled();
	void setDrawingEnabled(bool drawingEnabled);
	void draw(DrawingCanvas canvas, Mat target_xs, Mat target_ys);

private:
	int verbosity;
	bool drawingEnabled;
	int layerCount;
	int* layerShapes;
	vector<Neuron*>* layers;
	ErrorFunction* errorFunction;
	int maxIterations;
	float minError;
	float errorConvergenceThreshold;
	float weightConvergenceThreshold;
	int drawRate;
};