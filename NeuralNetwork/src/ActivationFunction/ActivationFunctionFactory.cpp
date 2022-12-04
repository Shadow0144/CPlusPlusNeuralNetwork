#include "ActivationFunction/ActivationFunctionFactory.h"

#include "ActivationFunction/ActivationFunction.h"

#include "ActivationFunction/IdentityFunction.h"
#include "ActivationFunction/ReLUFunction.h"
#include "ActivationFunction/AbsoluteReLUFunction.h"
#include "ActivationFunction/CReLUFunction.h"
#include "ActivationFunction/ELUFunction.h"
#include "ActivationFunction/SELUFunction.h"
#include "ActivationFunction/GELUFunction.h"
#include "ActivationFunction/LeakyReLUFunction.h"
#include "ActivationFunction/PReLUFunction.h"
#include "ActivationFunction/ReLU6Function.h"
#include "ActivationFunction/ReLUnFunction.h"
#include "ActivationFunction/SoftplusFunction.h"
#include "ActivationFunction/ExponentialFunction.h"
#include "ActivationFunction/QuadraticFunction.h"
#include "ActivationFunction/SigmoidFunction.h"
#include "ActivationFunction/TanhFunction.h"
#include "ActivationFunction/HardSigmoidFunction.h"
#include "ActivationFunction/SoftsignFunction.h"
#include "ActivationFunction/SwishFunction.h"

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
				activationFunction = new ELUFunction();
			}
			else
			{
				activationFunction = new ELUFunction(additionalParameters[ELUFunction::ALPHA]);
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
				activationFunction = new LeakyReLUFunction();
			}
			else
			{
				activationFunction = new LeakyReLUFunction(additionalParameters[LeakyReLUFunction::A]);
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
					activationFunction = new PReLUFunction(((int)(additionalParameters[PReLUFunction::NUM_UNITS])));
				}
				else
				{
					activationFunction = new PReLUFunction(((int)(additionalParameters[PReLUFunction::NUM_UNITS],
																  additionalParameters[PReLUFunction::A])));
				}
			}
			break;
		case ActivationFunctionType::ReLU6:
			activationFunction = new ReLU6Function();
			break;
		case ActivationFunctionType::ReLUn:
			if (additionalParameters.find(ReLUnFunction::N) == additionalParameters.end())
			{
				activationFunction = new ReLUnFunction();
			}
			else
			{
				activationFunction = new ReLUnFunction(additionalParameters[ReLUnFunction::N]);
			}
			break;
		case ActivationFunctionType::Softplus:
			if (additionalParameters.find(SoftplusFunction::K) == additionalParameters.end())
			{
				activationFunction = new SoftplusFunction();
			}
			else
			{
				activationFunction = new SoftplusFunction(additionalParameters[SoftplusFunction::K]);
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