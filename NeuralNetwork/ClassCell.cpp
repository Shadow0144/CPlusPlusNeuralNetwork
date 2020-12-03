#include "ClassCell.h"

ClassCell::ClassCell()
{

}

ClassCell::~ClassCell()
{

}

void ClassCell::draw(ImDrawList* canvas, ImVec2 origin, double scale)
{
	/*// Calculate the drawing space parameters
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

	double accuracy = 0;
	xt::xarray<size_t> predictedIndices = convertToIndices(predicted);
	xt::xarray<size_t> actualIndices = convertToIndices(actual);

	canvas->AddRectFilled(topLeft, bottomRight, EXTRA_LIGHT_GRAY);
	canvas->AddRect(topLeft, bottomRight, BLACK);

	float midX = bottomRight.x - ((bottomRight.x - topLeft.x) / 2.0f);
	canvas->AddText(ImGui::GetFont(), TEXT_SIZE, ImVec2(midX - (TEXT_SIZE * 5), topLeft.y + (TEXT_BUFFER / 2.0f)), BLACK, "Predicted");
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
			int predictedIndex = predictedIndices(index);
			int actualIndex = actualIndices(index);
			canvas->AddTriangleFilled(cellTopLeft, cellTopRight, cellBottomLeft, classColors[predictedIndex]);
			canvas->AddTriangleFilled(cellBottomRight, cellBottomLeft, cellTopRight, classColors[actualIndex]);
			canvas->AddLine(cellBottomLeft, cellTopRight, GRAY);
			if (predictedIndices(index) == actualIndices(index))
			{
				canvas->AddRect(cellTopLeft, cellBottomRight, RIGHT_COLOR);
				accuracy++;
			}
			else
			{
				canvas->AddRect(cellTopLeft, cellBottomRight, WRONG_COLOR);
			}
			index++;
		}
	}

	const int N = actualIndices.shape()[0];
	for (; index < N; index++)
	{
		if (predictedIndices(index) == actualIndices(index))
		{
			accuracy++;
		}
		else { }
	}
	accuracy /= N;
	string accuracyString = " Accuracy: " + to_string(accuracy);
	canvas->AddText(ImGui::GetFont(), TEXT_SIZE, topLeft, BLACK, accuracyString.c_str());*/
}