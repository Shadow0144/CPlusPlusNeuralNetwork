#include "DotProductFunction.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

DotProductFunction::DotProductFunction()
{

}

void DotProductFunction::feedForward()
{

}

void DotProductFunction::backPropagate()
{

}

void DotProductFunction::draw(DrawingCanvas canvas)
{
	const Scalar black(0, 0, 0);
	const Scalar white(255, 255, 255);
	const int line_sq_length = 8;
	Point start(canvas.offset.x - line_sq_length, canvas.offset.y + line_sq_length);
	Point end(canvas.offset.x + line_sq_length, canvas.offset.y - line_sq_length);
	rectangle(canvas.canvas, start, end, white, -1, LINE_8);
	line(canvas.canvas, start, end, black, 1, LINE_8);
}