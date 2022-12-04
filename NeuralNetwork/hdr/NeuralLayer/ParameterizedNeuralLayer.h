#pragma once

#include "NeuralLayer/NeuralLayer.h"
#include "ParameterSet.h"

class ParameterizedNeuralLayer : public NeuralLayer
{
public:
	virtual ~ParameterizedNeuralLayer() = 0; // This class is not intended to be directly instantiated

	double getRegularizationLoss(double lambda1, double lambda2) const;

	virtual void saveParameters(std::string fileName);
	virtual void loadParameters(std::string fileName);

	// Used by some optimizers
	virtual void substituteParameters(Optimizer* optimizer);
	virtual void restoreParameters(Optimizer* optimizer);

protected:
	ParameterizedNeuralLayer(NeuralLayer* parent);

	ParameterSet weights;
};