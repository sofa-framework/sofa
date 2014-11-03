#ifndef COMPLIANT_CompliantJacobiPreconditioner_H
#define COMPLIANT_CompliantJacobiPreconditioner_H

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

class SOFA_Compliant_API CompliantJacobiPreconditioner : public BasePreconditioner
{

  public:

    SOFA_ABSTRACT_CLASS(CompliantJacobiPreconditioner, BasePreconditioner);

    typedef AssembledSystem::real Real;

    CompliantJacobiPreconditioner();

    virtual void compute( const AssembledSystem::mat& H );
    virtual void apply( AssembledSystem::vec& res, const AssembledSystem::vec& v );

  protected:


    AssembledSystem::vec m_diagonal_inv;

};

}
}
}

#endif
 
