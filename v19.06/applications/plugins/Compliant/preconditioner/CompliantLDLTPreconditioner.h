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
 * @author Matthieu Nesme
*/

class SOFA_Compliant_API CompliantLDLTPreconditioner : public BasePreconditioner
{

  public:

    SOFA_ABSTRACT_CLASS(CompliantLDLTPreconditioner, BasePreconditioner);

    CompliantLDLTPreconditioner();

    void compute( const AssembledSystem::rmat& H ) override;
    void apply( AssembledSystem::vec& res, const AssembledSystem::vec& v ) override;

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
 
