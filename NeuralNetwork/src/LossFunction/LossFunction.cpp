#include "LossFunction/LossFunction.h"

#include "NeuralLayer/NeuralLayer.h"

void LossFunction::checkForOptimizedGradient(NeuralLayer* finalLayer)
{
	finalLayer->useSimplifiedGradient(false);
};