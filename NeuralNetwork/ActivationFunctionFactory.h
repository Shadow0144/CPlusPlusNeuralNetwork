#pragma once

#include "NeuralNetwork.h"

#include <map>

class ActivationFunction;

class ActivationFunctionFactory
{
public:
	static ActivationFunction* getNewActivationFunction(ActivationFunctionType functionType, std::map<string, double> additionalParameters);
};