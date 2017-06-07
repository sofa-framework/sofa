#include "CoulombConstraint.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {

using namespace sofa::defaulttype;

SOFA_COMPLIANT_CONSTRAINT_CPP(CoulombConstraintBase)


static int CoulombConstraintClass = 
    core::RegisterObject("standard coulomb constraint")
    
#ifndef SOFA_FLOAT
    .add< CoulombConstraint< Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
    .add< CoulombConstraint< Vec3fTypes > >()
#endif
    ;

// TODO register in the factory in case we want to use it manually in
// the graph

#ifndef SOFA_FLOAT
template struct SOFA_Compliant_API CoulombConstraint<Vec3dTypes>;
template struct SOFA_Compliant_API CoulombConstraint<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template struct SOFA_Compliant_API CoulombConstraint<Vec3fTypes>;
template struct SOFA_Compliant_API CoulombConstraint<Vec6fTypes>;
#endif





}
}
}

