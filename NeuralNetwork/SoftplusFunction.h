#pragma once

#include "Function.h"

using namespace cv;

// Softplus / SmoothReLU
class SoftplusFunction : public Function
{
public:
	SoftplusFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat lastInput, Mat errors);
	bool hasBias();
	void draw(DrawingCanvas canvas);

	float getK();
	void setK(float k);

	int numOutputs();

private:
	Mat lastOutput;

	float k = 1.0f; // Sharpness coefficient
};