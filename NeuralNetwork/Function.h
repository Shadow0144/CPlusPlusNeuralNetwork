#pragma once

#include "DrawingCanvas.h"
#include <opencv2/core.hpp>

using namespace cv;

class Function
{
public:
	virtual void feedForward() = 0;
	virtual void backPropagate() = 0;
	virtual void draw(DrawingCanvas canvas) = 0;
private:
	// Nothing
};