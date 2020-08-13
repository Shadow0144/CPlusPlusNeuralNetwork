#include "NeuralLayer.h"

const double NeuralLayer::RADIUS = 40;
const double NeuralLayer::DIAMETER = RADIUS * 2;
const double NeuralLayer::NEURON_SPACING = 20;

double NeuralLayer::getLayerWidth(size_t numUnits, double scale)
{
	return ((((DIAMETER + NEURON_SPACING) * numUnits) + NEURON_SPACING) * scale);
}

double NeuralLayer::getNeuronX(double originX, double layerWidth, int i, double scale)
{
	return (originX - (layerWidth * 0.5) + (((DIAMETER + NEURON_SPACING) * i) * scale));
}