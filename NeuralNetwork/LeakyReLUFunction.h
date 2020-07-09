#pragma once

#include "Function.h"

using namespace cv;

// Leaky ReLU / Parametric ReLU
class LeakyReLUFunction : public Function
{
public:
	LeakyReLUFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat lastInput, Mat errors);
	bool hasBias();
	void draw(DrawingCanvas canvas);

	float getA();
	void setA(float a);

	int numOutputs();

private:
	Mat lastOutput;

	float a = 0.01f; // Leak coefficient
};