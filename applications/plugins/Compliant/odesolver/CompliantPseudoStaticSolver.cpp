#include "CompliantPseudoStaticSolver.inl"

#include <sofa/core/ObjectFactory.h>


#include <Compliant/odesolver/CompliantImplicitSolver.h>
#include <Compliant/odesolver/ConstantCompliantImplicitSolver.h>


namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(CompliantPseudoStaticSolver)
int CompliantPseudoStaticSolverClass = core::RegisterObject("Iterative quasi-static solver")
        .add< CompliantPseudoStaticSolver<CompliantImplicitSolver> >(true)
        .add< CompliantPseudoStaticSolver<ConstantCompliantImplicitSolver> >()
        ;

template class SOFA_Compliant_API CompliantPseudoStaticSolver<CompliantImplicitSolver>;
template class SOFA_Compliant_API CompliantPseudoStaticSolver<ConstantCompliantImplicitSolver>;



}
}
}
