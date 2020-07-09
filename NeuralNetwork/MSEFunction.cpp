#include "MSEFunction.h"

float MSEFunction::getError(Mat predicted, Mat actual)
{
	int n = predicted.rows;
	Mat errors = Mat(1, n, CV_32FC1, cv::sum(predicted - actual));
	pow(errors, 2.0, errors);
	errors /= (2 * n);
	float error = ((float)(cv::sum(errors)[0]));
	return error;
}

Mat MSEFunction::getDerivativeOfError(Mat predicted, Mat actual)
{
	int exampleCount = predicted.rows;
	Mat errors = Mat(1, exampleCount, CV_32FC1, cv::sum(predicted - actual));
	return errors;
}