#include "ParameterSet.h"
#include "NeuralNetwork.h"

#include <iostream>

using namespace std;

ParameterSet::ParameterSet()
{

}

Mat ParameterSet::getParameters() 
{ 
	return parameters;
}

void ParameterSet::setParametersRandom(int parameterCount)
{
	parameters = Mat(Size(parameterCount, 1), CV_32FC1);

	RNG rng = RNG();
	cv::Mat mean = cv::Mat::zeros(1, 1, CV_64FC1);
	cv::Mat sigma = cv::Mat::ones(1, 1, CV_64FC1);
	rng.fill(parameters, cv::RNG::NORMAL, mean, sigma);
}

void ParameterSet::setParametersZero(int parameterCount)
{
	parameters = Mat::zeros(Size(parameterCount, 1), CV_32FC1);
}

void ParameterSet::setParametersOne(int parameterCount)
{
	parameters = Mat::ones(Size(parameterCount, 1), CV_32FC1);
}

void ParameterSet::setDeltaParameters(Mat deltaParameters)
{
	this->deltaParameters = deltaParameters;
}

void ParameterSet::applyDeltaParameters()
{
	parameters += deltaParameters;
}