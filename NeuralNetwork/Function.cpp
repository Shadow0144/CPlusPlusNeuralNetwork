#include "Function.h"

double Function::applyBackProgate()
{
	weights.applyDeltaParameters();
	return weights.getDeltaParameters().cwiseAbs().sum(); // Return the sum of how much the parameters have changed
}

void Function::draw(NetworkVisualizer canvas)
{
	/*const Scalar BLACK(0, 0, 0);
	const int DARK_GRAY = 50;
	const Scalar WHITE(255, 255, 255);
	const int DOT_LENGTH = 4;

	Point start(canvas.offset.x - DRAW_LEN, canvas.offset.y + DRAW_LEN);
	Point end(canvas.offset.x + DRAW_LEN, canvas.offset.y - DRAW_LEN);

	rectangle(canvas.canvas, start, end, WHITE, -1, LINE_8);

	Point zero_x_left(canvas.offset.x - DRAW_LEN, canvas.offset.y);
	Point zero_x_right(canvas.offset.x + DRAW_LEN, canvas.offset.y);
	LineIterator itX(canvas.canvas, zero_x_left, zero_x_right, LINE_8);
	Point zero_y_base(canvas.offset.x, canvas.offset.y + DRAW_LEN);
	Point zero_y_top(canvas.offset.x, canvas.offset.y - DRAW_LEN);
	LineIterator itY(canvas.canvas, zero_y_base, zero_y_top, LINE_8);
	for (int i = 0; i < itX.count; i++, itX++, itY++)
	{
		if (i % DOT_LENGTH != 0)
		{
			(*itX)[0] = DARK_GRAY;
			(*itX)[1] = DARK_GRAY;
			(*itX)[2] = DARK_GRAY;
			(*itY)[0] = DARK_GRAY;
			(*itY)[1] = DARK_GRAY;
			(*itY)[2] = DARK_GRAY;
		}
		else { }
	}*/
}