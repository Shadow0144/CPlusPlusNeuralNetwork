#pragma once

#include <vector>
#include "Neuron.h"

class NeuralNetwork
{
public:
	NeuralNetwork();

private:
	std::vector<Neuron> layers;
};