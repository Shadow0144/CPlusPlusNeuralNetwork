#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#include "imgui.h"
#pragma warning(pop)

class NetworkVisualizer;

class ImageClassVisualizer
{
public:
	ImageClassVisualizer(NetworkVisualizer* visualizer, int rows, int cols, ImColor* classColors);
	~ImageClassVisualizer();

	xt::xarray<size_t> convertToIndices(const xt::xarray<double>& predicted);

	void draw(ImDrawList* canvas, const xt::xarray<double>& predicted, const xt::xarray<double>& actual);

private:
	NetworkVisualizer* visualizer;
	int rows;
	int cols;
	ImColor* classColors;
};