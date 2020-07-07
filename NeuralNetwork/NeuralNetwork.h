#pragma once

#include <vector>
#include "Neuron.h"

using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork(int layerCount, int* layerShapes, ActivationFunction* layerFunctions);
	~NeuralNetwork();

	Mat feedForward(Mat input);
	bool backPropagate(Mat xs, Mat yHats);

	float MSE(Mat ys, Mat yHats);

	void draw(DrawingCanvas canvas, Mat target_xs, Mat target_ys);

private:
	int layerCount;
	int* layerShapes;
	vector<Neuron*>* layers;
};