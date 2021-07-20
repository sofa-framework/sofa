#include <sofa/component/linearsolver/preconditioner/RotationMatrixSystem.h>
#include <sofa/linearalgebra/RotationMatrix.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsolver::preconditioner
{
    using sofa::linearalgebra::RotationMatrix;
    using sofa::linearalgebra::FullVector;
    template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API RotationMatrixSystem< RotationMatrix<SReal>, FullVector<SReal> >;

    int RotationMatrixSystemClass = core::RegisterObject("Rotation matrix warpping the main linear system")
        .add<RotationMatrixSystem< RotationMatrix<SReal>, FullVector<SReal> > >(true);
}
