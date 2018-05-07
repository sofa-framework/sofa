#ifndef COMPLIANT_IncompleteCholeskyPreconditioner_H
#define COMPLIANT_IncompleteCholeskyPreconditioner_H

#include "BasePreconditioner.h"

#include <unsupported/Eigen/IterativeSolvers>

namespace sofa {
namespace component {
namespace linearsolver {

/**
 *
 *  Linear system preconditioner based on an Incomplete Cholesky factorization
 *
 * @author Matthieu Nesme
 *
*/

class SOFA_Compliant_API IncompleteCholeskyPreconditioner : public BasePreconditioner
{

  public:

    SOFA_ABSTRACT_CLASS(IncompleteCholeskyPreconditioner, BasePreconditioner);

    IncompleteCholeskyPreconditioner();

    typedef AssembledSystem::real real;
    typedef AssembledSystem::vec vec;
    typedef AssembledSystem::rmat rmat;
    typedef AssembledSystem::cmat cmat;

    virtual void reinit();

    virtual void compute( const rmat& H );
    virtual void apply( vec& res, const vec& v );

    Data<bool> d_constant; ///< reuse first factorization
    Data<real> d_shift; ///< initial shift

  protected:

    bool m_factorized;


    /// Incomplete Cholesky decomposition is always set for cmat
    Eigen::IncompleteCholesky< real > preconditioner;

};

}
}
}

#endif
 
