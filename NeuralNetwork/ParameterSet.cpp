#include "ParameterSet.h"
#include "NeuralNetwork.h"

#include <iostream>

using namespace std;

ParameterSet::ParameterSet()
{

}

MatrixXd ParameterSet::getParameters()
{ 
	return parameters;
}

void ParameterSet::setParametersRandom(int parameterCount)
{
	parameters = MatrixXd::Random(parameterCount, 1);
}

void ParameterSet::setParametersZero(int parameterCount)
{
	parameters = MatrixXd::Zero(parameterCount, 1);
}

void ParameterSet::setParametersOne(int parameterCount)
{
	parameters = MatrixXd::Ones(parameterCount, 1);
}

MatrixXd ParameterSet::getDeltaParameters()
{
	return deltaParameters;
}

void ParameterSet::setDeltaParameters(MatrixXd deltaParameters)
{
	this->deltaParameters = deltaParameters;
}

void ParameterSet::applyDeltaParameters()
{
	parameters -= deltaParameters;
}