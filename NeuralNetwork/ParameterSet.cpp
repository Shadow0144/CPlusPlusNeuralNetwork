#include "ParameterSet.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <time.h>

using namespace std;

ParameterSet::ParameterSet()
{
	srand((unsigned int)time(0));
}

MatrixXd ParameterSet::getParameters()
{ 
	return parameters;
}

void ParameterSet::setParametersRandom(int parameterCount)
{
	parameters = MatrixXd::Random(parameterCount, 1);
}

void ParameterSet::setParametersRandom(int inputCount, int outputCount)
{
	parameters = MatrixXd::Random(inputCount, outputCount);
}

void ParameterSet::setParametersZero(int parameterCount)
{
	parameters = MatrixXd::Zero(parameterCount, 1);
}

void ParameterSet::setParametersZero(int inputCount, int outputCount)
{
	parameters = MatrixXd::Zero(inputCount, outputCount);
}

void ParameterSet::setParametersOne(int parameterCount)
{
	parameters = MatrixXd::Ones(parameterCount, 1);
}

void ParameterSet::setParametersOne(int inputCount, int outputCount)
{
	parameters = MatrixXd::Ones(inputCount, outputCount);
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