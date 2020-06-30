#pragma once

#include "Function.h"

using namespace cv;

class IdentityFunction : public Function
{
public:
	IdentityFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat lastInput, Mat error);
	void draw(DrawingCanvas canvas);
};
