#include "CoulombConstraint.inl"

namespace sofa {
namespace component {
namespace linearsolver {

using namespace sofa::defaulttype; 

// TODO register in the factory in case we want to use it manually in
// the graph

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API CoulombConstraint<Vec3dTypes>;
template class SOFA_Compliant_API CoulombConstraint<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API CoulombConstraint<Vec3fTypes>;
template class SOFA_Compliant_API CoulombConstraint<Vec6fTypes>;
#endif

}
}
}

