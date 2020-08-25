#pragma once

#pragma warning( disable : 26451 )

#pragma warning(push, 0)
#include <Eigen/Core>
#include <Eigen/Dense>
#include "imgui.h"
#include <xtensor/xarray.hpp>
#pragma warning(pop)

#include "ParameterSet.h"

using namespace Eigen;

class Function
{
public:
	virtual xt::xarray<double> feedForward(xt::xarray<double> inputs) = 0;
	virtual xt::xarray<double> backPropagate(xt::xarray<double> sigmas) = 0;
	virtual double applyBackPropagate(); // Returns the sum of the change in the weights
	bool getHasBias() { return hasBias; }
	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale);
	ParameterSet getWeights() { return weights; }
protected:
	size_t numUnits;
	size_t numInputs;
	xt::xarray<double> lastInput;
	ParameterSet weights;
	const double DRAW_LEN = 16;

	bool hasBias = false;
	bool drawAxes = true;

	xt::xarray<double> dotProduct(xt::xarray<double> inputs);
	xt::xarray<double> denseBackpropagate(xt::xarray<double> sigmas);
	virtual xt::xarray<double> activationDerivative();

	virtual double activate(double z); // For drawing
	MatrixXd approximateBezier(MatrixXd points); // For drawing
	void approximateFunction(ImDrawList* canvas, ImVec2 origin, double scale); // For drawing

	const double ALPHA = 0.1; // Learning rate
};