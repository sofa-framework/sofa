#ifndef COMPLIANT_IncompleteCholeskyPreconditioner_H
#define COMPLIANT_IncompleteCholeskyPreconditioner_H

#include "BasePreconditioner.h"

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/IterativeSolvers>

//#include <unsupported/Eigen/ModuleHeader>

namespace sofa {
namespace component {
namespace linearsolver {

/**
 *
 * 
*/

class SOFA_Compliant_API IncompleteCholeskyPreconditioner : public BasePreconditioner
{

  public:

    SOFA_ABSTRACT_CLASS(IncompleteCholeskyPreconditioner, BasePreconditioner);

    virtual void compute( const AssembledSystem::mat& H );
    virtual void apply( AssembledSystem::vec& res, const AssembledSystem::vec& v );

  protected:


    /// the LDLT decomposition
    Eigen::IncompleteCholesky< AssembledSystem::cmat > preconditioner;

};

}
}
}

#endif
 
