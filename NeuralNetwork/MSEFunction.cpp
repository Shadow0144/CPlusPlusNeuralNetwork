#include "MSEFunction.h"

float MSEFunction::getError(Mat predicted, Mat actual)
{
	int n = predicted.rows;
	int m = predicted.cols;
	Mat errors = predicted - actual;
	pow(errors, 2.0, errors);
	errors /= (n * m);
	float error = ((float)(cv::sum(errors)[0]));
	return error;
}

Mat MSEFunction::getDerivativeOfError(Mat predicted, Mat actual)
{
	int n = predicted.rows;
	int m = predicted.cols;
	Mat sum = Mat(n, 1, CV_32FC1);
	cv::reduce((predicted - actual), sum, REDUCE_SUM, CV_32FC1);
	Mat errors = (2.0 / (n * m)) * sum;
	return errors;
}