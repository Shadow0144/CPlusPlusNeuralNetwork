#include "ParameterSet.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <time.h>

using namespace std;

ParameterSet::ParameterSet()
{
	srand((unsigned int)time(0));
	parameters = MatrixXd::Zero(0, 0);
	deltaParameters = MatrixXd::Zero(0, 0);
	batchSize = 0;
}

MatrixXd ParameterSet::getParameters()
{ 
	return parameters;
}

void ParameterSet::setParametersRandom(int inputCount, int outputCount)
{
	parameters = MatrixXd::Random(inputCount, outputCount);
	deltaParameters = MatrixXd::Zero(inputCount, outputCount);
	batchSize = 0;
}

void ParameterSet::setParametersZero(int inputCount, int outputCount)
{
	parameters = MatrixXd::Zero(inputCount, outputCount);
	deltaParameters = MatrixXd::Zero(inputCount, outputCount);
	batchSize = 0;
}

void ParameterSet::setParametersOne(int inputCount, int outputCount)
{
	parameters = MatrixXd::Ones(inputCount, outputCount);
	deltaParameters = MatrixXd::Zero(inputCount, outputCount);
	batchSize = 0;
}

MatrixXd ParameterSet::getDeltaParameters()
{
	return deltaParameters;
}

void ParameterSet::incrementDeltaParameters(MatrixXd deltaParameters)
{
	this->deltaParameters += deltaParameters;
	batchSize++;
}

void ParameterSet::applyDeltaParameters()
{
	parameters -= (deltaParameters / batchSize);
	deltaParameters = MatrixXd::Zero(parameters.rows(), parameters.cols());
	batchSize = 0;
}