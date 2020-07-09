#pragma once

#include "DrawingCanvas.h"
#include "ParameterSet.h"
#include <opencv2/core.hpp>

using namespace cv;

class Function
{
public:
	virtual Mat feedForward(Mat inputs) = 0;
	virtual Mat backPropagate(Mat lastInput, Mat errors) = 0;
	float applyBackProgate(); // Returns the sum of the change in the weights
	virtual bool hasBias() = 0;
	virtual void draw(DrawingCanvas canvas);
	ParameterSet getWeights() { return weights; }
	virtual int numOutputs() = 0;
protected:
	int numInputs = 0;
	ParameterSet weights;
	const int DRAW_LEN = 16;

	const float ALPHA = 0.1f; // Learning rate
};