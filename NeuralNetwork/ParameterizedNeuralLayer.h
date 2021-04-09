#pragma once

#include "NeuralLayer.h"
#include "ParameterSet.h"

class ParameterizedNeuralLayer : public NeuralLayer
{
public:
	double getRegularizationLoss(double lambda1, double lambda2) const;

	virtual void saveParameters(std::string fileName);
	virtual void loadParameters(std::string fileName);

	// Used by some optimizers
	virtual void substituteParameters(Optimizer* optimizer);
	virtual void restoreParameters(Optimizer* optimizer);

protected:
	ParameterSet weights;
};