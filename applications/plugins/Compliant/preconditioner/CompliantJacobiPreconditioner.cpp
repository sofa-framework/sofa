#include "CompliantJacobiPreconditioner.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {


SOFA_DECL_CLASS(CompliantJacobiPreconditioner);
int CompliantJacobiPreconditionerClass = core::RegisterObject("Jacobi preconditioner").add< CompliantJacobiPreconditioner >();


CompliantJacobiPreconditioner::CompliantJacobiPreconditioner()
    : BasePreconditioner()
{

}

void CompliantJacobiPreconditioner::compute( const AssembledSystem::mat& H )
{
    m_diagonal_inv = H.diagonal();
    for( AssembledSystem::mat::Index i=0 ; i<H.rows() ; ++i )
        m_diagonal_inv.coeffRef(i) = std::abs(m_diagonal_inv.coeff(i)) < std::numeric_limits<Real>::epsilon() ? Real(0) : Real(1) / m_diagonal_inv.coeff(i);
}

void CompliantJacobiPreconditioner::apply( AssembledSystem::vec& res, const AssembledSystem::vec& v )
{
//    std::cerr<<SOFA_CLASS_METHOD<<std::endl;
    res.resize( v.size() );
    res = m_diagonal_inv.asDiagonal() * v;
}

}
}
}

 
