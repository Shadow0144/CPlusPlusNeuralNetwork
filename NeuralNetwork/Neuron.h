#pragma once

#include "Function.h"
#include "DrawingCanvas.h"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

enum class ActivationFunction
{
	WeightedDotProduct,
	ReLU,
	Sigmoid,
	Tanh
};

class Neuron
{
public:
	Neuron(ActivationFunction function);
	~Neuron();

	float feedForward(Mat input);
	float backPropagate(float error);

	void draw(DrawingCanvas canvas, int* previous_xs, int previous_count, int previous_y, bool output);

private:
	ActivationFunction functionType;
	Function* activationFunction;
};