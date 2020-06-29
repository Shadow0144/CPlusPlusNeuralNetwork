#pragma once

#include <opencv2/core.hpp>

using namespace cv;

class ParameterSet
{
public:
	ParameterSet();
	Mat getParameters() { return parameters; }
	void setParameters(int parameterCount);

private:
	Mat parameters;
	Mat deltaParameters;
};