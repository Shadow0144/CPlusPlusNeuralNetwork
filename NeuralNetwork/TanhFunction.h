#pragma once

#include "Function.h"

using namespace cv;

class TanhFunction : public Function
{
public:
	TanhFunction(int numInputs);

	Mat feedForward(Mat input);
	Mat backPropagate(Mat error);
	void draw(DrawingCanvas canvas);
};