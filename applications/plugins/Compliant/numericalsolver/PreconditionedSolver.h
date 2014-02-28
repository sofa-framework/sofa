#ifndef COMPLIANT_PRECONDITIONEDSOLVER_H
#define COMPLIANT_PRECONDITIONEDSOLVER_H

#include "../initCompliant.h"

namespace sofa {
namespace component {
namespace linearsolver {

class BasePreconditioner;
struct AssembledSystem;




// base class for preconditioned solvers
class SOFA_Compliant_API PreconditionedSolver {
  public:

    PreconditionedSolver();

    void getPreconditioner( core::objectmodel::BaseContext* );

  protected:

    typedef BasePreconditioner preconditioner_type;
    preconditioner_type* _preconditioner;


    struct Preconditioner {
        BasePreconditioner* preconditioner;const AssembledSystem& sys;

        Preconditioner(const AssembledSystem& sys, BasePreconditioner* p)
            : preconditioner(p)
            , sys(sys)
        {
            p->compute(sys.H);
        }

        mutable AssembledSystem::vec result;

        template<class Vec>
        const AssembledSystem::vec& operator()(const Vec& x) const {
                preconditioner->apply( result, x );
                result = sys.P.selfadjointView<Eigen::Upper>() * result;
                return result;
        }

    };


};

}
}
}

#endif

