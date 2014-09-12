#include "JacobiPreconditioner.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {


SOFA_DECL_CLASS(JacobiPreconditioner);
int JacobiPreconditionerClass = core::RegisterObject("Jacobi preconditioner").add< JacobiPreconditioner >();


JacobiPreconditioner::JacobiPreconditioner()
    : BasePreconditioner()
{

}

void JacobiPreconditioner::compute( const AssembledSystem::mat& H )
{
    m_diagonal_inv = H.diagonal();
    for( AssembledSystem::mat::Index i=0 ; i<H.rows() ; ++i )
        m_diagonal_inv.coeffRef(i) = std::abs(m_diagonal_inv.coeff(i)) < std::numeric_limits<Real>::epsilon() ? Real(0) : Real(1) / m_diagonal_inv.coeff(i);
}

void JacobiPreconditioner::apply( AssembledSystem::vec& res, const AssembledSystem::vec& v )
{
    res.resize( v.size() );
    res = m_diagonal_inv.asDiagonal() * v;
}

}
}
}

 
