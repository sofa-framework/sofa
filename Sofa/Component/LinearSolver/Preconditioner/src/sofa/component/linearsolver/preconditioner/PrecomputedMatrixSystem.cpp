#include <sofa/component/linearsolver/preconditioner/PrecomputedMatrixSystem.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsolver::preconditioner
{
    using sofa::linearalgebra::CompressedRowSparseMatrix;
    using sofa::linearalgebra::FullVector;
    template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API PrecomputedMatrixSystem< linearalgebra::CompressedRowSparseMatrix<SReal>, FullVector<SReal> >;

    int PrecomputedMatrixSystemClass = core::RegisterObject("Matrix system")
        .add<PrecomputedMatrixSystem< CompressedRowSparseMatrix<SReal>, FullVector<SReal> > >(true);
}
