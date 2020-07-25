#pragma once

#pragma warning(push, 0)
#include <Eigen/Core>
#pragma warning(pop)

#include "imgui.h"

using namespace Eigen;

class ClassifierVisualizer
{
public:
	ClassifierVisualizer(int rows, int cols, ImColor* classColors);
	~ClassifierVisualizer();

	MatrixXi convertToIndices(MatrixXd matrix);

	void draw(ImDrawList* canvas, MatrixXd predicted, MatrixXd actual);

private:
	int rows;
	int cols;
	ImColor* classColors;
};