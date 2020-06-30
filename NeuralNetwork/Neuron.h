#pragma once

#include "Function.h"
#include "DrawingCanvas.h"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

enum class ActivationFunction
{
	Identity,
	WeightedDotProduct,
	ReLU,
	Sigmoid,
	Tanh
};

class Neuron
{
public:
	Neuron(ActivationFunction function, vector<Neuron*>* parents);
	~Neuron();

	Mat feedForward(Mat input);
	Mat backPropagate(Mat errors);
	void applyBackPropagate();

	void draw(DrawingCanvas canvas, bool output);

private:
	ActivationFunction functionType;
	Function* activationFunction;
	vector<Neuron*>* parents;
	int parentCount;
	int inputCount;
	vector<Neuron*>* children;
	int childCount;

	struct DrawingParameters
	{
		Point center;
	};
	DrawingParameters drawingParameters;

	Mat lastInput;
	Mat result; // Results of feedforward

	void addChild(Neuron* child);
};