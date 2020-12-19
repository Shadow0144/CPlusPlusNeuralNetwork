#pragma once

#include "NeuralNetwork.h"

class ActivationFunction;

class ActivationFunctionFactory
{
public:
	static ActivationFunction* getNewActivationFunction(ActivationFunctionType functionType);
};