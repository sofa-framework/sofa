#include "PythonMultiMapping.inl"

#include <sofa/core/MultiMapping.inl>
#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace mapping {


    
using namespace defaulttype;

// Register in the Factory
int PythonMultiMappingClass = core::RegisterObject("Arbitrary Python mapping")

.add< PythonMultiMapping< Rigid3Types, Vec6Types > >()            
.add< PythonMultiMapping< Rigid3Types, Vec3Types > >()        
.add< PythonMultiMapping< Rigid3Types, Vec1Types > >()

.add< PythonMultiMapping< Vec6Types, Rigid3Types > >()
.add< PythonMultiMapping< Vec3Types, Rigid3Types > >()
.add< PythonMultiMapping< Vec1Types, Rigid3Types > >()

    
.add< PythonMultiMapping< Vec6Types, Vec3Types > >()
.add< PythonMultiMapping< Vec3Types, Vec3Types > >()
.add< PythonMultiMapping< Vec1Types, Vec3Types > >()
    
.add< PythonMultiMapping< Vec6Types, Vec1Types > >()
.add< PythonMultiMapping< Vec3Types, Vec1Types > >()
.add< PythonMultiMapping< Vec1Types, Vec1Types > >()
//.add< PythonMultiMapping< Vec1Types, Rigid3Types > >()   // really? Needing some explanations

;

template class SOFA_Compliant_API PythonMultiMapping<  Rigid3Types, Vec6Types >;            
template class SOFA_Compliant_API PythonMultiMapping<  Rigid3Types, Vec3Types >;        
template class SOFA_Compliant_API PythonMultiMapping<  Rigid3Types, Vec1Types >;

template class SOFA_Compliant_API PythonMultiMapping<  Vec6Types, Rigid3Types >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec3Types, Rigid3Types >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec1Types, Rigid3Types >;

template class SOFA_Compliant_API PythonMultiMapping<  Vec6Types, Vec3Types >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec3Types, Vec3Types >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec1Types, Vec3Types >;

template class SOFA_Compliant_API PythonMultiMapping<  Vec6Types, Vec1Types >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec3Types, Vec1Types >;
template class SOFA_Compliant_API PythonMultiMapping<  Vec1Types, Vec1Types >;
//template class SOFA_Compliant_API PythonMultiMapping<  Vec1Types, Rigid3Types >;



    

}
}
}
