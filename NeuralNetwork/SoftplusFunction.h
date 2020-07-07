#pragma once

#include "Function.h"

using namespace cv;

class SoftplusFunction : public Function
{
public:
	SoftplusFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat lastInput, Mat errors);
	bool hasBias();
	void draw(DrawingCanvas canvas);

private:
	Mat lastOutput;
};