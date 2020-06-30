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
	parameters = Mat(parameterCount, 1, CV_32FC1);

	RNG rng = RNG(cv::getCPUTickCount());
	Scalar mean = 0.0f;
	Scalar sigma = 1.0f;
	rng.fill(parameters, RNG::NORMAL, mean, sigma);
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
	parameters -= deltaParameters;
}