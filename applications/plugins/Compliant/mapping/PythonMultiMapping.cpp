#include "PythonMultiMapping.h"

#include <sofa/core/MultiMapping.inl>
#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace mapping {

SOFA_DECL_CLASS(PythonMultiMapping)

using namespace defaulttype;

// Register in the Factory
const int PythonMultiMappingClass = core::RegisterObject("Arbitrary Python mapping")

#ifndef SOFA_FLOAT
.add< PythonMultiMapping< Vec6dTypes, Vec1dTypes > >()
.add< PythonMultiMapping< Vec3dTypes, Vec1dTypes > >()
.add< PythonMultiMapping< Vec1dTypes, Vec1dTypes > >()
//.add< PythonMultiMapping< Vec1dTypes, Rigid3dTypes > >()   // really? Needing some explanations
#endif
#ifndef SOFA_DOUBLE
.add< PythonMultiMapping< Vec6fTypes, Vec1fTypes > >()
.add< PythonMultiMapping< Vec3fTypes, Vec1fTypes > >()
.add< PythonMultiMapping< Vec1fTypes, Vec1fTypes > >()
//.add< PythonMultiMapping< Vec1fTypes, Rigid3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API PythonMultiMapping<  Vec6dTypes, Vec1dTypes >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec3dTypes, Vec1dTypes >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec1dTypes, Vec1dTypes >;
//template class SOFA_Compliant_API PythonMultiMapping<  Vec1dTypes, Rigid3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API PythonMultiMapping< Vec6fTypes, Vec1fTypes >;
template class SOFA_Compliant_API PythonMultiMapping< Vec3fTypes, Vec1fTypes >;
template class SOFA_Compliant_API PythonMultiMapping< Vec1fTypes, Vec1fTypes >;
//template class SOFA_Compliant_API PythonMultiMapping< Vec1fTypes, Rigid3fTypes >;
#endif



}
}
}
