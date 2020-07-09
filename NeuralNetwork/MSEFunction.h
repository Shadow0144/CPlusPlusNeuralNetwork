#pragma once

#include <opencv2/core.hpp>

#include "ErrorFunction.h"

using namespace cv;

class MSEFunction : public ErrorFunction
{
public:
	float getError(Mat predicted, Mat actual);
	Mat getDerivativeOfError(Mat predicted, Mat actual);
};