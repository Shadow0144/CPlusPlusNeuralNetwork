#include "Function.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void Function::applyBackProgate()
{
	weights.applyDeltaParameters();
}

void Function::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);
	const int dark_gray = 50;
	const Scalar white(255, 255, 255);
	const int dot_length = 4;

	Point start(canvas.offset.x - draw_len, canvas.offset.y + draw_len);
	Point end(canvas.offset.x + draw_len, canvas.offset.y - draw_len);

	rectangle(canvas.canvas, start, end, white, -1, LINE_8);

	Point zero_x_left(canvas.offset.x - draw_len, canvas.offset.y);
	Point zero_x_right(canvas.offset.x + draw_len, canvas.offset.y);
	LineIterator itX(canvas.canvas, zero_x_left, zero_x_right, LINE_8);
	Point zero_y_base(canvas.offset.x, canvas.offset.y + draw_len);
	Point zero_y_top(canvas.offset.x, canvas.offset.y - draw_len);
	LineIterator itY(canvas.canvas, zero_y_base, zero_y_top, LINE_8);
	for (int i = 0; i < itX.count; i++, itX++, itY++)
	{
		if (i % dot_length != 0)
		{
			(*itX)[0] = dark_gray;
			(*itX)[1] = dark_gray;
			(*itX)[2] = dark_gray;
			(*itY)[0] = dark_gray;
			(*itY)[1] = dark_gray;
			(*itY)[2] = dark_gray;
		}
		else { }
	}
}