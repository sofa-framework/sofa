#include "CoulombConstraint.inl"
#include <sofa/defaulttype/VecTypes.h>

namespace sofa {
namespace component {
namespace linearsolver {

using namespace sofa::defaulttype;

SOFA_COMPLIANT_CONSTRAINT_CPP(CoulombConstraintBase)

// TODO register in the factory in case we want to use it manually in
// the graph

template struct SOFA_Compliant_API CoulombConstraint<Vec3Types>;
template struct SOFA_Compliant_API CoulombConstraint<Vec6Types>;


}
}
}

