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

	float getK() { return k; }
	void setK(float k) { this->k = k; }

private:
	Mat lastOutput;

	float k = 1.0f; // Sharpness coefficient
};