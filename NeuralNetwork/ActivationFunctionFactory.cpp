#include "ActivationFunctionFactory.h"

#include "ActivationFunction.h"

#include "IdentityFunction.h"
#include "ReLUFunction.h"
#include "AbsoluteReLUFunction.h"
#include "CReLUFunction.h"
#include "ELUFunction.h"
#include "SELUFunction.h"
#include "GELUFunction.h"
#include "LeakyReLUFunction.h"
#include "PReLUFunction.h"
#include "ReLU6Function.h"
#include "ReLUnFunction.h"
#include "SoftplusFunction.h"
#include "ExponentialFunction.h"
#include "QuadraticFunction.h"
#include "SigmoidFunction.h"
#include "TanhFunction.h"
#include "HardSigmoidFunction.h"
#include "SoftsignFunction.h"
#include "SwishFunction.h"

#pragma warning(push, 0)
#include <exception>
#pragma warning(pop)

using namespace std;

ActivationFunction* ActivationFunctionFactory::getNewActivationFunction(ActivationFunctionType functionType, std::map<string, double> additionalParameters)
{
	ActivationFunction* activationFunction;
	switch (functionType)
	{
		case ActivationFunctionType::Identity:
			activationFunction = new IdentityFunction();
			break;
		case ActivationFunctionType::ReLU:
			activationFunction = new ReLUFunction();
			break;
		case ActivationFunctionType::AbsoluteReLU:
			activationFunction = new AbsoluteReLUFunction();
			break;
		case ActivationFunctionType::CReLU:
			activationFunction = new CReLUFunction();
			break;
		case ActivationFunctionType::ELU:
			if (additionalParameters.find(ELUFunction::ALPHA) == additionalParameters.end())
			{
				activationFunction = new ELUFunction(additionalParameters[ELUFunction::ALPHA]);
			}
			else
			{
				activationFunction = new ELUFunction();
			}
			break;
		case ActivationFunctionType::SELU:
			activationFunction = new SELUFunction();
			break;
		case ActivationFunctionType::GELU:
			activationFunction = new GELUFunction();
			break;
		case ActivationFunctionType::LeakyReLU:
			if (additionalParameters.find(LeakyReLUFunction::A) == additionalParameters.end())
			{
				activationFunction = new LeakyReLUFunction(additionalParameters[LeakyReLUFunction::A]);
			}
			else
			{
				activationFunction = new LeakyReLUFunction();
			}
			break;
		case ActivationFunctionType::PReLU:
			if (additionalParameters.find(PReLUFunction::NUM_UNITS) == additionalParameters.end())
			{
				throw std::invalid_argument(std::string("Missing required parameter: ") +
					"PReLUFunction::NUM_UNITS" + " (\"" + PReLUFunction::NUM_UNITS + "\")");
			}
			else
			{
				if (additionalParameters.find(PReLUFunction::A) == additionalParameters.end())
				{
					activationFunction = new PReLUFunction(((int)(additionalParameters[PReLUFunction::NUM_UNITS],
																  additionalParameters[PReLUFunction::A])));
				}
				else
				{
					activationFunction = new PReLUFunction(((int)(additionalParameters[PReLUFunction::NUM_UNITS])));
				}
			}
			break;
		case ActivationFunctionType::ReLU6:
			activationFunction = new ReLU6Function();
			break;
		case ActivationFunctionType::ReLUn:
			if (additionalParameters.find(ReLUnFunction::N) == additionalParameters.end())
			{
				activationFunction = new ReLUnFunction(additionalParameters[ReLUnFunction::N]);
			}
			else
			{
				activationFunction = new ReLUnFunction();
			}
			break;
		case ActivationFunctionType::Softplus:
			if (additionalParameters.find(SoftplusFunction::K) == additionalParameters.end())
			{
				activationFunction = new SoftplusFunction(additionalParameters[SoftplusFunction::K]);
			}
			else
			{
				activationFunction = new SoftplusFunction();
			}
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
		case ActivationFunctionType::Swish:
			activationFunction = new SwishFunction();
			break;
		default:
			activationFunction = new IdentityFunction();
			break;
	}
	return activationFunction;
}