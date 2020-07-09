#pragma once

#include <opencv2/core.hpp>

using namespace cv;

class ParameterSet
{
public:
	ParameterSet();
	Mat getParameters();
	void setParametersRandom(int parameterCount);
	void setParametersZero(int parameterCount);
	void setParametersOne(int parameterCount);
	Mat getDeltaParameters();
	void setDeltaParameters(Mat deltaParameters);
	void applyDeltaParameters();

private:
	Mat parameters;
	Mat deltaParameters;
};