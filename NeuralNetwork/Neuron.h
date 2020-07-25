#pragma once

#include "Function.h"
#include "imgui.h"
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
	Neuron(ActivationFunction function, vector<Neuron*>* parents, int inputCount, int outputCount = -1);
	~Neuron();

	MatrixXd feedForward(MatrixXd input);
	MatrixXd backPropagate(MatrixXd errors);
	double applyBackPropagate();

	int getNumOutputs();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output);

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

	ImVec2 position;

	void addChild(Neuron* child);

	friend class NetworkVisualizer;
};