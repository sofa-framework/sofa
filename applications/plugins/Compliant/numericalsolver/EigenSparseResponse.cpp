#include "EigenSparseResponse.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {




template class SOFA_Compliant_API EigenSparseResponse< LDLTSparseLinearSolver, LDLTSparseLinearSolver::UpLo >;
int LDLTResponseClass = core::RegisterObject("A sparse LDLT Cholesky factorization of the response matrix.").add< LDLTResponse >();

template class SOFA_Compliant_API EigenSparseResponse< LLTSparseLinearSolver, LLTSparseLinearSolver::UpLo >;
int LLTResponseClass = core::RegisterObject("A sparse LLT Cholesky factorization of the response matrix.").add< LLTResponse >();

template class SOFA_Compliant_API EigenSparseResponse< LUSparseLinearSolver, 0 >;
int LUResponseClass = core::RegisterObject("A sparse LU factorization of the response matrix.").add< LUResponse >();


/////////////////////////////////////////////


template class SOFA_Compliant_API EigenSparseIterativeResponse< CGSparseLinearSolver, true >;
int EigenCGResponseClass = core::RegisterObject("Conjugate Gradient solve of the response matrix.").add< EigenCGResponse >();


}
}
}
