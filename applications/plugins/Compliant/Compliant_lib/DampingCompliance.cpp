#include "DampingCompliance.h"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

using namespace defaulttype;

// Register in the Factory
int DampingComplianceClass = core::RegisterObject("Damping Compliance")
#ifndef SOFA_FLOAT
	.add< DampingCompliance< Vec6dTypes > >(true)
#endif
#ifndef SOFA_DOUBLE
	.add< DampingCompliance< Vec6fTypes > >(true)
#endif
	;

SOFA_DECL_CLASS(DampingCompilance)


template class SOFA_Compliant_API DampingCompliance<Vec6fTypes>;
template class SOFA_Compliant_API DampingCompliance<Vec6dTypes>;

}
}
}

