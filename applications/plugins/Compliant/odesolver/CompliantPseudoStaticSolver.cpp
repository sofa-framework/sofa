#include "CompliantPseudoStaticSolver.inl"

#include <sofa/core/ObjectFactory.h>


#include <Compliant/odesolver/CompliantImplicitSolver.h>
#include <Compliant/odesolver/ConstantCompliantImplicitSolver.h>


namespace sofa {
namespace component {
namespace odesolver {



int CompliantPseudoStaticSolverClass = core::RegisterObject("Iterative quasi-static solver")
        .add< CompliantPseudoStaticSolver<CompliantImplicitSolver> >(true)
        .add< CompliantPseudoStaticSolver<ConstantCompliantImplicitSolver> >()
        ;

template class SOFA_Compliant_API CompliantPseudoStaticSolver<CompliantImplicitSolver>;
template class SOFA_Compliant_API CompliantPseudoStaticSolver<ConstantCompliantImplicitSolver>;


class SOFA_Compliant_API ConstantCompliantPseudoStaticSolver:public CompliantPseudoStaticSolver<ConstantCompliantImplicitSolver>
{};

int ConstantCompliantPseudoStaticSolverClass = core::RegisterObject("Iterative quasi-static solver with constant system")
        .add< ConstantCompliantPseudoStaticSolver >(true)
        ;


}
}
}
