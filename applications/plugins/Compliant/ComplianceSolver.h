#ifndef SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#define SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#include "initCompliant.h"
#include "BaseCompliance.h"
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include "../ModelHierarchies/EigenSparseSquareMatrix.h"
#include "../ModelHierarchies/EigenSparseRectangularMatrix.h"
#include <plugins/ModelHierarchies/EigenVector.h>

namespace sofa
{
using helper::vector;

namespace component
{

namespace odesolver
{

/** Solver using a regularized KKT matrix.
  The compliance values are used to regularize the system.


  Inspired from Servin,Lacoursière,Melin, Interactive Simulation of Elastic Deformable Materials,  http://www.ep.liu.se/ecp/019/005/ecp01905.pdf

  Generalized to a tunable implicit integration scheme. The equation is:
  \f$ \left( \begin{array}{cc} M & -J^T \\ LJ & \frac{1}{l} C \end{arrax}\right)
      \left( \begin{array}{c} \delta v & \bar \lambda \end{array}\right)
    = \left( \begin{array}{c} f & - \frac{1}{l} (\phi +(d+\alpha h) \dot \phi)  \end{array}\right) \f$
    where \f$ M \f$ the mass matrix, \f$ \phi \f$ is the constraint violation, \f$ J \f$ the constraint Jacobian matrix,
    \f$ C \f$ is the compliance matrix (i.e. inverse of constraint stiffness), \f$ l=\alpha(h \beta + d) \f$ is a term related to implicit integration and constraint damping,
     \f$ \alpha \f$  and \f$ \beta \f$  define the integration scheme for a time step \f$ h \f$:

      \f$ \delta v = h.M^{-1}.(\alpha f_{n+1} + (1-\alpha) f_n)  \f$,   where \f$ \alpha \f$ is the implicit velocity factor.

      \f$ \delta x = h.(\beta v_{n+1} + (1-\beta) v_n)  \f$,   where \f$ \beta \f$ is the implicit position factor.

      \f$ \lambda \f$ is the average constraint force, consistently with to the implicit velocity integration.

  The equation is solved using a Shur complement: \f$ ( JM^{-1}J^T + C ) \f$


*/
class SOFA_Compliant_API ComplianceSolver  : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(ComplianceSolver, sofa::core::behavior::OdeSolver);

    virtual void bwdInit();
    virtual void solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);


protected:
    ComplianceSolver();

    typedef Eigen::DynamicSparseMatrix<SReal, Eigen::RowMajor> Matrix;
    typedef linearsolver::EigenVector<SReal>  Vector;
    typedef core::behavior::BaseMechanicalState MechanicalState;
    typedef core::behavior::BaseCompliance Compliance;
    typedef core::BaseMapping Mapping;

    Matrix matM;      ///< mass matrix
    Matrix matJ;      ///< concatenation of the constraint Jacobians
    Matrix matC;      ///< compliance matrix used to regularize the system
    Vector vecF;      ///< top of the right-hand term: forces
    Vector vecPhi;    ///< bottom of the right-hand term: constraint corrections

    Data<SReal>  implicitVelocity; ///< the \f$ \alpha \f$ parameter of the integration scheme
    Data<SReal>  implicitPosition; ///< the \f$ \beta  \f$ parameter of the integration scheme



    typedef enum { COMPUTE_SIZE, MATRIX_ASSEMBLY, VECTOR_ASSEMBLY } Pass;  ///< Symbols of operations to execute by the visitor

    /** Visitor used to perform the assembly of M, C, J.
      Proceeds in several passes:<ol>
      <li> the first pass counts the size of the matrices </li>
      <li> the matrices and the right-hand side vector are assembled in the second pass </li>
      </ol>
      */
    struct MatrixAssemblyVisitor: public simulation::MechanicalVisitor
    {
        ComplianceSolver* solver;
        core::ComplianceParams cparams;
        unsigned sizeM; ///< size of the mass matrix
        unsigned sizeC; ///< size of the compliance matrix, number of scalar constraints

        MatrixAssemblyVisitor(const core::ComplianceParams* params, ComplianceSolver* s)
            : simulation::MechanicalVisitor(params)
            , solver(s)
            , cparams(*params)
            , sizeM(0)
            , sizeC(0)
            , pass(COMPUTE_SIZE)
        {}


        Pass pass;  ///< symbol to represent the current operation
        /// Set the operation to execute during the next traversal
        MatrixAssemblyVisitor& operator() ( Pass p ) { pass =p; return *this; }

        virtual Visitor::Result processNodeTopDown(simulation::Node* node);
        virtual void processNodeBottomUp(simulation::Node* node);

        std::map<MechanicalState*, unsigned> m_offset;  ///< Start index of independent DOFs in the mass matrix
        std::map<Compliance*, unsigned>      c_offset;  ///< Start index of compliances in the compliance matrix
        std::stack<Matrix> jStack;                      ///< Stack of jacobian matrices to push/pop during the traversal

        /// Return a rectangular matrix (cols>rows), with (offset-1) null columns, then the (rows*rows) identity, then null columns.
        /// This is used to shift a "local" matrix to the global indices of an assembly matrix.
        Matrix createShiftMatrix( unsigned rows, unsigned cols, unsigned offset );

        /// Converts a BaseMatrix to the matrix type used here.
        Matrix toMatrix( const defaulttype::BaseMatrix* );
    };



};

}
}
}

#endif // SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
