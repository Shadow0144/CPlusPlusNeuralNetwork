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
	
	xt::xarray<size_t> convertToIndices(const xt::xarray<double>& predicted);

	void draw(ImDrawList* canvas, const xt::xarray<double>& predicted, const xt::xarray<double>& actual);

private:
	NetworkVisualizer* visualizer;
	int rows;
	int cols;
	ImColor* classColors;
	const int MAX_ROWS = 20;
	const int MAX_COLS = 5;
};