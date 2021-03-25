#pragma once

#pragma warning(push, 0)
#include "imgui.h"
#include <xtensor/xarray.hpp>
#pragma warning(pop)

#include "ActivationFunction.h"

class Optimizer;

class NeuralLayer
{
public:
	~NeuralLayer();

	virtual void addChildren(NeuralLayer* children);
	size_t getNumUnits();

	virtual xt::xarray<double> feedForward(const xt::xarray<double>& input) = 0;
	virtual xt::xarray<double> feedForwardTrain(const xt::xarray<double>& input);
	// Returns the new sigmas and updates gradient with the local gradient
	virtual xt::xarray<double> getGradient(const xt::xarray<double>& sigmas, Optimizer* optimizer) = 0;
	virtual double applyBackPropagate() = 0;

	virtual std::vector<size_t> getOutputShape();

	virtual void saveParameters(std::string fileName);
	virtual void loadParameters(std::string fileName);

	// Used by some optimizers
	virtual void substituteParameters(Optimizer* optimizer);
	virtual void restoreParameters(Optimizer* optimizer);

	// Drawing constants
	const static double DRAW_LEN;// = 16.0;
	const static double RERESCALE;// = 0.75;
	const static double SHIFT;// = 16.0;
	const static double RADIUS;// = 40;
	const static double DIAMETER;// = RADIUS * 2;
	const static double NEURON_SPACING;// = 20;
	const static double LINE_LENGTH;// = 15;
	const static double WEIGHT_RADIUS;// = 10;
	const static double BIAS_OFFSET_X;// = 40;
	const static double BIAS_OFFSET_Y;// = -52;
	const static double BIAS_FONT_SIZE;// = 24;
	const static double BIAS_WIDTH;// = 20;
	const static double BIAS_HEIGHT;// = BIAS_FONT_SIZE;
	const static double BIAS_TEXT_X;// = 4;
	const static double BIAS_TEXT_Y;// = 20;
	const static ImColor BLACK;
	const static ImColor GRAY;
	const static ImColor LIGHT_GRAY;
	const static ImColor VERY_LIGHT_GRAY;
	const static ImColor WHITE;

	static double getLayerWidth(size_t numUnits, double scale); // Drawing helper function
	static double getNeuronX(double originX, double layerWidth, int i, double scale); // Drawing helper function

	virtual void draw(ImDrawList* canvas, ImVec2 origin, double scale, bool output) = 0;

protected:
	NeuralLayer* parent = nullptr;
	NeuralLayer* children = nullptr;

	ActivationFunctionType functionType;
	ActivationFunction* activationFunction = nullptr;

	std::vector<size_t> inputShape;
	xt::xarray<double> lastInput;
	xt::xarray<double> lastOutput;

	const double ALPHA = 0.01; // Learning rate

	size_t numUnits; // For drawing
	ImVec2 position; // For drawing

	xt::xarray<double> addBiasToInput(const xt::xarray<double>& input);

	void drawFunctionBackground(ImDrawList* canvas, ImVec2 origin, double scale, bool drawAxes);
	void drawConversionFunctionBackground(ImDrawList* canvas, ImVec2 origin, double scale, bool drawAxes);
};