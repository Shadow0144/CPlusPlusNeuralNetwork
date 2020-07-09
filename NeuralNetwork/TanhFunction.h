#pragma once

#include "Function.h"

using namespace cv;

class TanhFunction : public Function
{
public:
	TanhFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat lastInput, Mat errors);
	bool hasBias();
	void draw(DrawingCanvas canvas);

	int numOutputs();

private:
	Mat lastOutput;
};