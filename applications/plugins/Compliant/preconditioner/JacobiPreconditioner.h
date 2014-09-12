#ifndef COMPLIANT_JACOBIPRECONDITIONER_H
#define COMPLIANT_JACOBIPRECONDITIONER_H

#include "BasePreconditioner.h"

namespace sofa {
namespace component {
namespace linearsolver {

/**
 * 
 *  Jacobi/diagonal Linear system preconditioner
 *  efficient for diagonal dominant matrices
 * 
*/

class SOFA_Compliant_API JacobiPreconditioner : public BasePreconditioner
{

  public:

    SOFA_ABSTRACT_CLASS(JacobiPreconditioner, BasePreconditioner);

    typedef AssembledSystem::real Real;

    JacobiPreconditioner();

    virtual void compute( const AssembledSystem::mat& H );
    virtual void apply( AssembledSystem::vec& res, const AssembledSystem::vec& v );

  protected:


    AssembledSystem::vec m_diagonal_inv;

};

}
}
}

#endif
 
