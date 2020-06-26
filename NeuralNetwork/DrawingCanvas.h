#pragma once

#include <opencv2/core.hpp>

using namespace cv;

struct DrawingCanvas
{
public:
	Mat canvas;
	Point offset;
	float scale;
};