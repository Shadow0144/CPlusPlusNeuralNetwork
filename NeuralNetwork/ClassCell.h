#pragma once

#pragma warning(push, 0)
#include <xtensor/xarray.hpp>
#include "imgui.h"
#pragma warning(pop)

class ClassCell
{
public:
	ClassCell();
	~ClassCell();

	void draw(ImDrawList* canvas, ImVec2 origin, double scale);

private:
	int classCount;
	int classNum;
	ImColor rightColor;
	ImColor* wrongColors;
};