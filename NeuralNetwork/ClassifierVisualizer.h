#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#include "imgui.h"
#pragma warning(pop)

class NetworkVisualizer;

class ClassifierVisualizer
{
public:
	ClassifierVisualizer(NetworkVisualizer* visualizer, int rows, int cols, ImColor* classColors);
	~ClassifierVisualizer();
	
	xt::xarray<size_t> convertToIndices(xt::xarray<double> predicted);

	void draw(ImDrawList* canvas, xt::xarray<double> predicted, xt::xarray<double> actual);

private:
	NetworkVisualizer* visualizer;
	int rows;
	int cols;
	ImColor* classColors;
};