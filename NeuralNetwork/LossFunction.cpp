#include "LossFunction.h"

#include "NeuralLayer.h"

void LossFunction::checkForOptimizedGradient(NeuralLayer* finalLayer)
{
	finalLayer->useSimplifiedGradient(false);
};