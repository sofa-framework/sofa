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
    .add< DampingCompliance< Vec2dTypes > >()
    .add< DampingCompliance< Vec1dTypes > >()
#endif
#ifndef SOFA_DOUBLE
    .add< DampingCompliance< Vec6fTypes > >(true)
    .add< DampingCompliance< Vec2fTypes > >()
    .add< DampingCompliance< Vec1fTypes > >()
#endif
	;

SOFA_DECL_CLASS(DampingCompilance)

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API DampingCompliance<Vec6dTypes>;
template class SOFA_Compliant_API DampingCompliance<Vec2dTypes>;
template class SOFA_Compliant_API DampingCompliance<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API DampingCompliance<Vec6fTypes>;
template class SOFA_Compliant_API DampingCompliance<Vec2fTypes>;
template class SOFA_Compliant_API DampingCompliance<Vec1fTypes>;
#endif
}
}
}

