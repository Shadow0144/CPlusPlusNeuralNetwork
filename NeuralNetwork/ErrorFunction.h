#pragma once

#include <opencv2/core.hpp>

using namespace cv;

class ErrorFunction
{
public:
	virtual float getError(Mat predicted, Mat actual) = 0;
	virtual Mat getDerivativeOfError(Mat predicted, Mat actual) = 0;
};