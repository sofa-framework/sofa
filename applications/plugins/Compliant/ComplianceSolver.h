#ifndef SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#define SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#include "initCompliant.h"
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/linearsolver/EigenSparseSquareMatrix.h>
#include <sofa/component/linearsolver/EigenVector.h>
#include <set>

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

  We generalize it to a tunable implicit integration scheme:
      \f[ \begin{array}{ccc}
    \Delta v &=& h.M^{-1}.(\alpha f_{n+1} + (1-\alpha) f_n)  \\
    \Delta x &=& h.(\beta v_{n+1} + (1-\beta) v_n)
    \end{array} \f]
    where \f$ h \f$ is the time step, \f$ \alpha \f$ is the implicit velocity factor, and \f$ \beta \f$ is the implicit position factor.

    The corresponding dynamic equation is:
  \f[ \left( \begin{array}{cc} \frac{1}{h} M & -J^T \\
                               J & \frac{1}{l} C \end{array}\right)
      \left( \begin{array}{c} \Delta v \\ \bar\lambda \end{array}\right)
    = \left( \begin{array}{c} f \\ - \frac{1}{l} (\phi +(d+\alpha h) \dot \phi)  \end{array}\right) \f]
    where \f$ M \f$ is the mass matrix, \f$ \phi \f$ is the constraint violation, \f$ J \f$ the constraint Jacobian matrix,
    \f$ C \f$ is the compliance matrix (i.e. inverse of constraint stiffness), \f$ l=\alpha(h \beta + d) \f$ is a term related to implicit integration and constraint damping, and
      \f$ \bar\lambda \f$ is the average constraint forces, consistently with the implicit velocity integration.

  A Shur complement is used to compute the constraint forces, then these are added to the external forces to obtain the final velocity increment,
  and the positions are updated according to the implicit scheme:

  \f[ \begin{array}{ccc}
   ( hJPM^{-1}PJ^T + \frac{1}{l}C ) \bar\lambda &=& -\frac{1}{l} (\phi + (d+h\alpha)\dot\phi ) - h J M^{-1} f \\
                                 \Delta v  &=&  h P M^{-1}( f + J^T \bar\lambda ) \\
                                 \Delta x  &=&  h( v + \beta \Delta v )
  \end{array} \f]
where \f$ P \f$ is the projection matrix corresponding to the projective constraints applied to the independent DOFs.

\sa \ref sofa::core::behavior::BaseCompliance



*/
class SOFA_Compliant_API ComplianceSolver  : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(ComplianceSolver, sofa::core::behavior::OdeSolver);


    /** Set up the matrices and vectors of the equation system, call solveEquation() to solve the system, then applies the results
      You probably need not overload this method.
      */
    virtual void solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);


protected:
    ComplianceSolver();
    virtual ~ComplianceSolver() {}

    typedef Eigen::SparseMatrix<SReal, Eigen::RowMajor> SMatrix;
    typedef linearsolver::EigenVector<SReal>            VectorSofa;
    typedef Eigen::Matrix<SReal, Eigen::Dynamic, 1>     VectorEigen;
    typedef core::behavior::BaseMechanicalState         MechanicalState;
    typedef core::BaseMapping                           Mapping;

    // Equation system
    SMatrix matM;      ///< mass matrix
    SMatrix matP;      ///< projection matrix used to apply simple boundary conditions like fixed points
    SMatrix matJ;      ///< concatenation of the constraint Jacobians
    SMatrix matC;      ///< compliance matrix used to regularize the system
    VectorSofa vecF;   ///< top of the right-hand term: forces
    VectorSofa vecPhi; ///< bottom of the right-hand term: constraint corrections

    /** Solve the following equation system on matrices M,P,J,C and vectors f,phi.
     The method computes \f$f\f$ by solving the equations:
    \f$
    \begin{array}{ccc}
    (J.P.M^{-1}.P.J^T + C) \lambda &=&  \phi - J M^{-1} f_e \\
    f &=& f_e + J^T \lambda
    \end{array}
    \f$

    where matrices \f$M,P,J,C\f$ are members matM, matP, matJ, matC respectively, and vectors \f$\phi\f$ and \f$f\f$ are members vecPhi and vecF, respectively.
    The values of these terms depend on the integration scheme, and they are given in the detailed documentation of this class.
    Vector \f$ \lambda \f$ is a temporary vector declared in the method.
    Vector vecF initially contains the value of \f$ f_e \f$ , and it is replaced with \f$ f_e + J^T \lambda \f$.

     The base implementation uses a direct Cholesky solver. It can be overloaded to apply other linear equation solvers.
     */
    virtual void solveEquation();

public:
    Data<SReal>  implicitVelocity; ///< the \f$ \alpha \f$ parameter of the integration scheme
    Data<SReal>  implicitPosition; ///< the \f$ \beta  \f$ parameter of the integration scheme
    Data<bool>   verbose;          ///< print a lot of debug info

protected:

    typedef enum { COMPUTE_SIZE, DO_SYSTEM_ASSEMBLY, DISTRIBUTE_SOLUTION } Pass;  ///< Symbols of operations to execute by the visitor

    /** Visitor used to perform the assembly of M, C, J.
      Proceeds in several passes:<ol>
      <li> the first pass counts the size of the matrices </li>
      <li> the matrices and the right-hand side vector are assembled in the second pass </li>
      </ol>
      */
    struct MatrixAssemblyVisitor: public simulation::MechanicalVisitor
    {
        ComplianceSolver* solver;
        core::MechanicalParams cparams;
        unsigned sizeM; ///< size of the mass matrix
        unsigned sizeC; ///< size of the compliance matrix, number of scalar constraints
        std::set<core::behavior::BaseMechanicalState*> localDOFs;  ///< Mechanical DOFs in the range of the solver. This is used to discard others, such interaction mouse DOFs

        MatrixAssemblyVisitor(const core::MechanicalParams* params, ComplianceSolver* s)
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

        std::map<MechanicalState*, unsigned> m_offset;                 ///< Start index of independent DOFs in the mass matrix
        std::map<core::behavior::BaseForceField*, unsigned> c_offset;  ///< Start index of compliances in the compliance matrix
        std::map<MechanicalState*,SMatrix> jMap;                       ///< jacobian matrices of each mechanical state, with respect to the vector of all independent DOFs.

        /// Return a rectangular matrix (cols>rows), with (offset-1) null columns, then the (rows*rows) identity, then null columns.
        /// This is used to shift a "local" matrix to the global indices of an assembly matrix.
        SMatrix createShiftMatrix( unsigned rows, unsigned cols, unsigned offset );

        /// Return an identity matrix of the given size
        SMatrix createIdentityMatrix( unsigned size );

        /// Casts the matrix using a dynamic_cast. Crashes if the BaseMatrix* is not a SMatrix*
        const SMatrix& getSMatrix( const defaulttype::BaseMatrix* );
    };


    // sparse LDLT support (requires  SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD compile flag)
    typedef Eigen::SparseLDLT<Eigen::SparseMatrix<SReal>,Eigen::Cholmod>  SparseLDLT;  // process SparseMatrix, not DynamicSparseMatrix (not implemented in Cholmod)

    /// Compute the inverse of the matrix. The input matrix MUST be diagonal. The threshold parameter is currently unused.
    void inverseDiagonalMatrix( SMatrix& minv, const SMatrix& m, SReal threshold);
    SMatrix invM;      ///< inverse mass matrix used in the Schur complement


};

}
}
}

#endif // SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
