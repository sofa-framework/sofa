#ifndef SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#define SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#include "initCompliant.h"
#include "BaseCompliance.h"
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/common/Visitor.h>
#include "../ModelHierarchies/EigenSparseSquareMatrix.h"
#include "../ModelHierarchies/EigenSparseRectangularMatrix.h"

namespace sofa
{
using helper::vector;

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
    typedef core::behavior::BaseMechanicalState MechanicalState;
    typedef core::behavior::BaseCompliance Compliance;
    typedef core::BaseMapping Mapping;

    Matrix matM;      ///< the mass matrix
    Matrix matJ;      ///< the concatenation of the constraint Jacobians
    Matrix matC;      ///< the compliance matrix used to regularize the system

//    vector<MechanicalState::SPtr> states;  ///< The independent states
//    vector<Compliance::SPtr> compliances;  ///< All the compliances

    /** Visitor used to perform the assembly of M, C, and J.
      Proceeds in several passes:<ol>
      <li> the first pass counts the size of the matrices </li>
      <li> the matrices are then created </li>
      <li> the second pass performs the assembly by summing matrix products </li>
      </ol>
      */
    struct MatrixAssemblyVisitor: public simulation::Visitor
    {
        ComplianceSolver* solver;
        unsigned sizeM; ///< size of the mass matrix
        unsigned sizeC; ///< size of the compliance matrix, number of scalar constraints

        MatrixAssemblyVisitor(const core::ExecParams* params, ComplianceSolver* s): simulation::Visitor(params), solver(s), sizeM(0), sizeC(0), pass(1) {}
        virtual Visitor::Result processNodeTopDown(simulation::Node* node);
//        virtual void processNodeBottomUp(simulation::Node* node);

        unsigned pass;  ///< Counter to represent the current pass
        std::map<MechanicalState*, unsigned> m_offset;  ///< Start index of independent DOFs in the mass matrix
        std::map<Compliance*, unsigned>      c_offset;  ///< Start index of compliances in the compliance matrix
    };


};

}
}
}

#endif // SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
