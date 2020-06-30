#pragma once

#include "Function.h"

using namespace cv;

class DotProductFunction : public Function
{
public:
	DotProductFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat lastInput, Mat errors);
	void draw(DrawingCanvas canvas);
};