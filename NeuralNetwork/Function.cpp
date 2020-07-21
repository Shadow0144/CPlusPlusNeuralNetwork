#include "Function.h"

double Function::applyBackProgate()
{
	weights.applyDeltaParameters();
	return weights.getDeltaParameters().cwiseAbs().sum(); // Return the sum of how much the parameters have changed
}

MatrixXd Function::approximateBezier(MatrixXd points)
{
	//int pCount = points.rows();
	//int k = pCount - 1;

	/*MatrixXd T(pCount, pCount);
	for (int i = 0; i < pCount; i++)
	{
		for (int j = 0; j < pCount; j++)
		{
			double jk = ((double)j) / ((double)k);
			T(k-i, k-j) = pow(jk, i);
		}
	}*/

	MatrixXd T(4, 4);
	T << 1.0,       0.0,       0.0,        0.0,
		 1.0, 1.0 / 3.0, 1.0 / 9.0, 1.0 / 27.0,
		 1.0, 2.0 / 3.0, 4.0 / 9.0, 8.0 / 27.0,
		 1.0,       1.0,       1.0,        1.0;

	MatrixXd M(4, 4);
	M << 1,  0,  0,  0,
		-3,  3,  0,  0,
		 3, -6,  3,  0,
		-1,  3, -3,  1;

	/*MatrixXd M(7, 7);
	M <<     1,   0,   0,   0,   0,   0,   0,
			 -6,   6,   0,   0,   0,   0,   0,
			 15, -30,  15,   0,   0,   0,   0,
			-20,  60, -60, -20,   0,   0,   0,
			 15, -60,  90, -60,  15,   0,   0,
			 -6,  30, -60,  60, -30,   6,   0,
			  1,  -6,  15, -20,  15,  -6,   1;*/

	MatrixXd Mi = M.inverse();
	MatrixXd TT = T.transpose();
	MatrixXd A = Mi * (TT * T).inverse() * TT;

	return A * points;
}

void Function::draw(ImDrawList* canvas, ImVec2 origin, float scale)
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