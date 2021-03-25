#pragma once

#include "NeuralLayer.h"
#include "ParameterSet.h"

class ParameterizedNeuralLayer : public NeuralLayer
{
public:
	virtual void saveParameters(std::string fileName);
	virtual void loadParameters(std::string fileName);

	// Used by some optimizers
	virtual void substituteParameters(Optimizer* optimizer);
	virtual void restoreParameters(Optimizer* optimizer);

protected:
	ParameterSet weights;
};