#pragma once

class Function
{
public:
	virtual void feedForward() = 0;
	virtual void backPropagate() = 0;
private:
	// Nothing
};