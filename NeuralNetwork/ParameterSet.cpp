#include "ParameterSet.h"
#include "NeuralNetwork.h"

#include <iostream>

using namespace std;

ParameterSet::ParameterSet()
{

}

void ParameterSet::setParameters(int parameterCount)
{
	parameters = Mat(Size(1, parameterCount), CV_32FC1);

	RNG rng = RNG();
	cv::Mat mean = cv::Mat::zeros(1, 1, CV_64FC1);
	cv::Mat sigma = cv::Mat::ones(1, 1, CV_64FC1);
	rng.fill(parameters, cv::RNG::NORMAL, mean, sigma);
}