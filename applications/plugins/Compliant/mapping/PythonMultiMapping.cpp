#include "PythonMultiMapping.inl"

#include <sofa/core/MultiMapping.inl>
#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace mapping {


    
using namespace defaulttype;

// Register in the Factory
int PythonMultiMappingClass = core::RegisterObject("Arbitrary Python mapping")

#ifndef SOFA_FLOAT
.add< PythonMultiMapping< Rigid3dTypes, Vec6dTypes > >()            
.add< PythonMultiMapping< Rigid3dTypes, Vec3dTypes > >()        
.add< PythonMultiMapping< Rigid3dTypes, Vec1dTypes > >()

.add< PythonMultiMapping< Vec6dTypes, Rigid3dTypes > >()
.add< PythonMultiMapping< Vec3dTypes, Rigid3dTypes > >()
.add< PythonMultiMapping< Vec1dTypes, Rigid3dTypes > >()

    
.add< PythonMultiMapping< Vec6dTypes, Vec3dTypes > >()
.add< PythonMultiMapping< Vec3dTypes, Vec3dTypes > >()
.add< PythonMultiMapping< Vec1dTypes, Vec3dTypes > >()
    
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
template class SOFA_Compliant_API PythonMultiMapping<  Rigid3dTypes, Vec6dTypes >;            
template class SOFA_Compliant_API PythonMultiMapping<  Rigid3dTypes, Vec3dTypes >;        
template class SOFA_Compliant_API PythonMultiMapping<  Rigid3dTypes, Vec1dTypes >;

template class SOFA_Compliant_API PythonMultiMapping<  Vec6dTypes, Rigid3dTypes >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec3dTypes, Rigid3dTypes >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec1dTypes, Rigid3dTypes >;

template class SOFA_Compliant_API PythonMultiMapping<  Vec6dTypes, Vec3dTypes >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec3dTypes, Vec3dTypes >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec1dTypes, Vec3dTypes >;

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
