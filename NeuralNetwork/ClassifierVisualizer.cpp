#include "ClassifierVisualizer.h"
#include "NetworkVisualizer.h"

ClassifierVisualizer::ClassifierVisualizer(NetworkVisualizer* visualizer, int rows, int cols, ImColor* classColors)
{
	this->visualizer = visualizer;
	this->rows = rows;
	this->cols = cols;
	this->classColors = classColors;
}

ClassifierVisualizer::~ClassifierVisualizer()
{

}

xt::xarray<size_t> ClassifierVisualizer::convertToIndices(xt::xarray<double> results)
{
	xt::xstrided_slice_vector sv({ 0 });
	for (int i = 1; i < (results.dimension() - 1); i++)
	{
		sv.push_back(0);
	}
	sv.push_back(xt::all());
	size_t c = results.shape()[results.dimension() - 1];
	const size_t N = results.shape()[0];
	std::vector<size_t> shape = { N };
	xt::xarray<size_t> r(shape);
	for (int i = 0; i < N; i++)
	{
		int index = 0;
		sv[0] = i;
		double indexValue = xt::strided_view(results, sv)(0);
		for (int j = 1; j < c; j++)
		{
			double t = xt::strided_view(results, sv)(j);
			if (xt::strided_view(results, sv)(j) > indexValue)
			{
				indexValue = xt::strided_view(results, sv)(j);
				index = j;
			}
			else { }
		}
		r(i) = index;
	}
	return r;
}

void ClassifierVisualizer::draw(ImDrawList* canvas, xt::xarray<double> predicted, xt::xarray<double> actual)
{
	// Calculate the drawing space parameters
	ImVec2 winSize = visualizer->getWindowSize();
	const double WIDTH = winSize.x;
	const double HEIGHT = winSize.y;
	const double HALF_WIDTH = WIDTH * 0.5;
	const double HALF_HEIGHT = HEIGHT * 0.5;

	const float BUFFER = 20;
	const float ROW_SIZE = 15;
	const float COL_SIZE = 50;
	const float CELL_BUFFER = 2;
	const float TEXT_SIZE = 18;
	const float TEXT_BUFFER = 40;
	const ImColor BLACK(0.0f, 0.0f, 0.0f, 1.0f);
	const ImColor GRAY(0.5f, 0.5f, 0.5f, 1.0f);
	const ImColor EXTRA_LIGHT_GRAY(0.75f, 0.75f, 0.75f, 1.0f);
	const ImColor VERY_LIGHT_GRAY(0.8f, 0.8f, 0.8f, 1.0f);
	const ImColor RIGHT_COLOR(0.0f, 1.0f, 0.0f, 1.0f);
	const ImColor WRONG_COLOR(1.0f, 0.0f, 0.0f, 1.0f);

	ImVec2 bottomRight = ImVec2(WIDTH - BUFFER, HEIGHT - BUFFER);
	ImVec2 topLeft = ImVec2(
		bottomRight.x - (COL_SIZE * cols) - (2 * CELL_BUFFER), 
		bottomRight.y - (ROW_SIZE * rows) - (2 * CELL_BUFFER) - (2 * TEXT_BUFFER));

	xt::xarray<size_t> predictedIndices = convertToIndices(predicted);
	xt::xarray<size_t> actualIndices = convertToIndices(actual);

	canvas->AddRectFilled(topLeft, bottomRight, EXTRA_LIGHT_GRAY);
	canvas->AddRect(topLeft, bottomRight, BLACK);

	float midX = bottomRight.x - ((bottomRight.x - topLeft.x) / 2.0f);
	canvas->AddText(ImGui::GetFont(), TEXT_SIZE, ImVec2(midX, topLeft.y + (TEXT_BUFFER / 2.0f)), BLACK, "Predicted");
	canvas->AddText(ImGui::GetFont(), TEXT_SIZE, ImVec2(midX, bottomRight.y - TEXT_BUFFER), BLACK, "Actual");

	size_t index = 0;
	for (int i = 0; i < cols; i++)
	{
		float xl = CELL_BUFFER + topLeft.x + (i * COL_SIZE) + CELL_BUFFER;
		float xr = CELL_BUFFER + topLeft.x + ((i + 1) * COL_SIZE) - CELL_BUFFER;
		for (int j = 0; j < rows; j++)
		{
			float yt = CELL_BUFFER + topLeft.y + (j * ROW_SIZE) + CELL_BUFFER + TEXT_BUFFER;
			float yb = CELL_BUFFER + topLeft.y + ((j + 1) * ROW_SIZE) - CELL_BUFFER + TEXT_BUFFER;
			ImVec2 cellTopLeft = ImVec2(xl, yt);
			ImVec2 cellTopRight = ImVec2(xr, yt);
			ImVec2 cellBottomLeft = ImVec2(xl, yb);
			ImVec2 cellBottomRight = ImVec2(xr, yb);
			canvas->AddTriangleFilled(cellTopLeft, cellTopRight, cellBottomLeft, classColors[predictedIndices(index)]);
			canvas->AddTriangleFilled(cellBottomRight, cellBottomLeft, cellTopRight, classColors[actualIndices(index)]);
			canvas->AddLine(cellBottomLeft, cellTopRight, GRAY);
			if (predictedIndices(index) == actualIndices(index))
			{
				canvas->AddRect(cellTopLeft, cellBottomRight, RIGHT_COLOR);
			}
			else
			{
				canvas->AddRect(cellTopLeft, cellBottomRight, WRONG_COLOR);
			}
			index++;
		}
	}
}