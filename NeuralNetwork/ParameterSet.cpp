#include "ParameterSet.h"

ParameterSet::ParameterSet()
{

}

void ParameterSet::setParameters(int parameterCount)
{
	parameters = Mat(Size(1, parameterCount), CV_32FC1);
	randu(parameters, Scalar(-1.0), Scalar(1.0));
}