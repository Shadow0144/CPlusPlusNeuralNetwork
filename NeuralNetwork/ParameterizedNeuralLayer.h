#pragma once

#include "NeuralLayer.h"
#include "ParameterSet.h"

using namespace std;

class ParameterizedNeuralLayer : public NeuralLayer
{
public:
	virtual void saveParameters(std::string fileName);
	virtual void loadParameters(std::string fileName);

protected:
	ParameterSet weights;
};