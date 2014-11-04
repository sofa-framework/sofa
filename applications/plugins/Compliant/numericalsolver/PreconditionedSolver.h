#ifndef COMPLIANT_PRECONDITIONEDSOLVER_H
#define COMPLIANT_PRECONDITIONEDSOLVER_H

#include "../initCompliant.h"

namespace sofa {
namespace component {
namespace linearsolver {

class BasePreconditioner;
struct AssembledSystem;




/// Base class for preconditioned solvers
class SOFA_Compliant_API PreconditionedSolver {
  public:

    PreconditionedSolver();

    void getPreconditioner( core::objectmodel::BaseContext* );

  protected:

    typedef BasePreconditioner preconditioner_type;
    preconditioner_type* _preconditioner;


    struct Preconditioner {
        BasePreconditioner* preconditioner;
        const AssembledSystem& sys;

        Preconditioner(const AssembledSystem& sys, BasePreconditioner* p)
            : preconditioner(p)
            , sys(sys)
        {
            if( sys.isPIdentity )
                p->compute(sys.H);
            else
            {
                AssembledSystem::cmat identity(sys.H.rows(),sys.H.cols());
                identity.setIdentity();
                p->compute( sys.P.transpose()*sys.H*sys.P + identity * std::numeric_limits<SReal>::epsilon() );
            }
        }

        mutable AssembledSystem::vec result;

        template<class Vec>
        const AssembledSystem::vec& operator()(const Vec& x) const {
                preconditioner->apply( result, x );
                return result;
        }

    };


};

}
}
}

#endif

