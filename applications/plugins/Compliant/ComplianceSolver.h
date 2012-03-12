#ifndef SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#define SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#include "initCompliant.h"
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/common/Visitor.h>
#include "../ModelHierarchies/EigenSparseSquareMatrix.h"
#include "../ModelHierarchies/EigenSparseRectangularMatrix.h"

namespace sofa
{

namespace component
{

namespace odesolver
{

/** Solver using a regularized KKT matrix.
  The compliance values are used to regularize the system.

  The equation is solved using a Shur complement: \f$ ( JM^{-1}J^T + C )\lambda = c  \f$

  Inspired from Servin,Lacoursière,Melin, Interactive Simulation of Elastic Deformable Materials,  http://www.ep.liu.se/ecp/019/005/ecp01905.pdf
*/
class SOFA_Compliant_API ComplianceSolver  : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(ComplianceSolver, sofa::core::behavior::OdeSolver);

    virtual void solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);


protected:
    ComplianceSolver();

    typedef Eigen::DynamicSparseMatrix<SReal> Matrix;

    Matrix matM;      ///< the mass matrix
    Matrix matJ;      ///< the concatenation of the constraint Jacobians
    Matrix matC;      ///< the compliance matrix used to regularize the system

    struct ComputeMatrixSizesVisitor: public simulation::Visitor
    {
        ComputeMatrixSizesVisitor(const core::ExecParams* params): simulation::Visitor(params), sizeM(0), sizeJ(0) {}

        unsigned sizeM; ///< size of the mass matrix
        unsigned sizeJ; ///< number of rows of J (=number of scalar constraints). The number of columns is the same as M.

        virtual void processNodeBottomUp(simulation::Node* node);
    };

};

}
}
}

#endif // SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
