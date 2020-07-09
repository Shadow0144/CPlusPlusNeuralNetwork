#pragma once

#include "Function.h"

using namespace cv;

class LeakyReLUFunction : public Function
{
public:
	LeakyReLUFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat lastInput, Mat errors);
	bool hasBias();
	void draw(DrawingCanvas canvas);

	float getAlpha() { return alpha; }
	void setAlpha(float alpha) { this->alpha = alpha; }

private:
	Mat lastOutput;

	float alpha = 0.1f; // Leak coefficient
};