#pragma once

#include "ActivationFunction/ActivationFunction.h"

#pragma warning(push, 0)
#include <map>
#pragma warning(pop)

class ActivationFunctionFactory
{
public:
	static ActivationFunction* getNewActivationFunction(ActivationFunctionType functionType, std::map<std::string, double> additionalParameters);
};