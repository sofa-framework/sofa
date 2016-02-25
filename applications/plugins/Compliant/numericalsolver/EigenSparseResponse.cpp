#include "EigenSparseResponse.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {


template class SOFA_Compliant_API EigenSparseResponse< LDLTSparseLinearSolver, LDLTSparseLinearSolver::UpLo >;
SOFA_DECL_CLASS(LDLTResponse)
int LDLTResponseClass = core::RegisterObject("A sparse Cholesky factorization of the response matrix.").add< LDLTResponse >();


template class SOFA_Compliant_API EigenSparseResponse< LUSparseLinearSolver, 0 >;
SOFA_DECL_CLASS(LUResponse)
int LUResponseClass = core::RegisterObject("A sparse LU factorization of the response matrix.").add< LUResponse >();


/////////////////////////////////////////////


template class SOFA_Compliant_API EigenSparseIterativeResponse< CGSparseLinearSolver, true >;
SOFA_DECL_CLASS(EigenCGResponse)
int EigenCGResponseClass = core::RegisterObject("Conjugate Gradient solve of the response matrix.").add< EigenCGResponse >();


}
}
}
