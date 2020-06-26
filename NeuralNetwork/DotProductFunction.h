#pragma once

#include "Function.h"

using namespace cv;

class DotProductFunction : public Function
{
public:
	DotProductFunction();

	void feedForward();
	void backPropagate();
	void draw(DrawingCanvas canvas);
private:

};