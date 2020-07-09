#pragma once

#include "Function.h"

using namespace cv;

class ConvolutionFunction : public Function
{
public:
	ConvolutionFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat lastInput, Mat errors);
	bool hasBias();
	void draw(DrawingCanvas canvas);

	int numOutputs();
};