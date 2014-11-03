#ifndef COMPLIANT_CompliantLDLTPreconditioner_H
#define COMPLIANT_CompliantLDLTPreconditioner_H

#include "BasePreconditioner.h"
#include <Eigen/SparseCholesky>

namespace sofa {
namespace component {
namespace linearsolver {

/**
 * 
 *  Linear system preconditioner based on LDLT pre-factorization
 *  @warning in most cases this preconditioner is really ineficient and must be used with care!
 * 
*/

class SOFA_Compliant_API CompliantLDLTPreconditioner : public BasePreconditioner
{

  public:

    SOFA_ABSTRACT_CLASS(CompliantLDLTPreconditioner, BasePreconditioner);

    CompliantLDLTPreconditioner();

    virtual void compute( const AssembledSystem::mat& H );
    virtual void apply( AssembledSystem::vec& res, const AssembledSystem::vec& v );

  protected:

    /// is the LDLT already factorized?
    bool _factorized;

    /// the LDLT decomposition
    Eigen::SimplicialLDLT< AssembledSystem::cmat > preconditioner;

};

}
}
}

#endif
 
