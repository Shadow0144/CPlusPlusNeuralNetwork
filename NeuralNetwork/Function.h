#pragma once

#include "DrawingCanvas.h"
#include "ParameterSet.h"
#include <opencv2/core.hpp>

using namespace cv;

class Function
{
public:
	virtual Mat feedForward(Mat inputs) = 0;
	virtual Mat backPropagate(Mat lastInput, Mat errors) = 0;
	void applyBackProgate();
	virtual void draw(DrawingCanvas canvas);
protected:
	int numInputs;
	ParameterSet weights;
	const int draw_len = 16;

	const float ALPHA = 0.001f; // Learning rate
};