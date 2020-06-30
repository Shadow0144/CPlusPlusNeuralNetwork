#pragma once

#include "Function.h"

using namespace cv;

class TanhFunction : public Function
{
public:
	TanhFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat lastInput, Mat errors);
	void draw(DrawingCanvas canvas);

private:
	Mat lastOutput;
};