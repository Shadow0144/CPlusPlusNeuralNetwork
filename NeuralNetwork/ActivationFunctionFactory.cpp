#include "ActivationFunctionFactory.h"

#include "ActivationFunction.h"

#include "LinearFunction.h"
#include "ReLUFunction.h"
#include "AbsoluteReLUFunction.h"
//#include "CReLUFunction.h"
#include "ELUFunction.h"
#include "SELUFunction.h"
#include "GELUFunction.h"
#include "LeakyReLUFunction.h"
//#include "PReLUFunction.h"
#include "ReLU6Function.h"
#include "ReLUnFunction.h"
#include "SoftplusFunction.h"
#include "ExponentialFunction.h"
#include "QuadraticFunction.h"
#include "SigmoidFunction.h"
#include "TanhFunction.h"
#include "HardSigmoidFunction.h"
#include "SoftsignFunction.h"
//#include "SwishFunction.h"
//#include "MaxoutFunction.h"

ActivationFunction* ActivationFunctionFactory::getNewActivationFunction(ActivationFunctionType functionType)
{
	ActivationFunction* activationFunction;
	switch (functionType)
	{
		case ActivationFunctionType::None:
			activationFunction = new LinearFunction();
			break;
		case ActivationFunctionType::ReLU:
			activationFunction = new ReLUFunction();
			break;
		case ActivationFunctionType::AbsoluteReLU:
			activationFunction = new AbsoluteReLUFunction();
			break;
		/*case ActivationFunctionType::CReLU:
			activationFunction = new CReLUFunction();*/
			break;
		case ActivationFunctionType::ELU:
			activationFunction = new ELUFunction();
			break;
		case ActivationFunctionType::SELU:
			activationFunction = new SELUFunction();
			break;
		case ActivationFunctionType::GELU:
			activationFunction = new GELUFunction();
			break;
		case ActivationFunctionType::LeakyReLU:
			activationFunction = new LeakyReLUFunction();
			break;
		/*case ActivationFunctionType::PReLU:
			activationFunction = new PReLUFunction();*/
			break;
		case ActivationFunctionType::ReLU6:
			activationFunction = new ReLU6Function();
			break;
		case ActivationFunctionType::ReLUn:
			activationFunction = new ReLUnFunction();
			break;
		case ActivationFunctionType::Softplus:
			activationFunction = new SoftplusFunction();
			break;
		case ActivationFunctionType::Exponential:
			activationFunction = new ExponentialFunction();
			break;
		case ActivationFunctionType::Quadratic:
			activationFunction = new QuadraticFunction();
			break;
		case ActivationFunctionType::Sigmoid:
			activationFunction = new SigmoidFunction();
			break;
		case ActivationFunctionType::Tanh:
			activationFunction = new TanhFunction();
			break;
		case ActivationFunctionType::HardSigmoid:
			activationFunction = new HardSigmoidFunction();
			break;
		case ActivationFunctionType::Softsign:
			activationFunction = new SoftsignFunction();
			break;
		/*case ActivationFunctionType::Swish:
			activationFunction = new SwishFunction();
			break;*/
		/*case ActivationFunctionType::Maxout:
			activationFunction = new MaxoutFunction(5); // TODO
			break;*/
		default:
			activationFunction = new LinearFunction();
			break;
	}
	return activationFunction;
}