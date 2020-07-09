#pragma once

#include <vector>
#include "Neuron.h"
#include "ErrorFunction.h"

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(int layerCount, int* layerShapes, ActivationFunction* layerFunctions, ErrorFunction* errorFunction);
	~NeuralNetwork();

	Mat feedForward(Mat input);
	bool backPropagate(Mat xs, Mat yHats);

	float getError(Mat predicted, Mat actual);

	void draw(DrawingCanvas canvas, Mat target_xs, Mat target_ys);

private:
	int layerCount;
	int* layerShapes;
	vector<Neuron*>* layers;
	ErrorFunction* errorFunction;
};