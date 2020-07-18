#pragma once

#include "Function.h"
#include "NetworkVisualizer.h"
#include <vector>

using namespace std;

enum class ActivationFunction
{
	Identity,
	WeightedDotProduct,
	ReLU,
	LeakyReLU,
	Softplus,
	Sigmoid,
	Tanh,
	Softmax
};

class Neuron
{
public:
	Neuron(ActivationFunction function, vector<Neuron*>* parents);
	~Neuron();

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd errors);
	double applyBackPropagate();

	void draw(NetworkVisualizer canvas, bool output);

private:
	ActivationFunction functionType;
	Function* activationFunction;
	vector<Neuron*>* parents;
	int parentCount;
	int inputCount;
	vector<Neuron*>* children;
	int childCount;

	MatrixXd lastInput;
	MatrixXd result; // Results of feedforward

	void addChild(Neuron* child);

	friend class NetworkVisualizer;
};