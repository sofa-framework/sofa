#ifndef SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#define SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
#include "initCompliant.h"
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/linearsolver/EigenSparseSquareMatrix.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/EigenVector.h>
#include <set>
#include <Eigen/Dense>
typedef Eigen::MatrixXd DenseMatrix;

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
  \f[ \left( \begin{array}{cc} \frac{1}{h} PM & -PJ^T \\
                               J & \frac{1}{l} C \end{array}\right)
      \left( \begin{array}{c} \Delta v \\ \bar\lambda \end{array}\right)
    = \left( \begin{array}{c} Pf \\ - \frac{1}{l} (\phi +(d+\alpha h) \dot \phi)  \end{array}\right) \f]
    where \f$ M \f$ is the mass matrix, \f$ P \f$ is a projection matrix to impose boundary conditions on displacements (typically maintain fixed points), \f$ \phi \f$ is the constraint violation, \f$ J \f$ the constraint Jacobian matrix,
    \f$ C \f$ is the compliance matrix (i.e. inverse of constraint stiffness) used to soften the constraints, \f$ l=\alpha(h \beta + d) \f$ is a term related to implicit integration and constraint damping, and
      \f$ \bar\lambda \f$ is the average constraint forces, consistently with the implicit velocity integration.

      The system is singular due to matrix \f$ P \f$, however we can use \f$ P M^{-1}P \f$ as inverse mass matrix to compute Schur complements.

 In the default implementation, a Schur complement is used to compute the constraint forces, then these are added to the external forces to obtain the final velocity increment,
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
    virtual void init();


    /** Set up the matrices and vectors of the equation system, call solveEquation() to solve the system, then applies the results
      You probably need not overload this method.
      */
    virtual void solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);


protected:
    ComplianceSolver();
    virtual ~ComplianceSolver() {}

public:
    typedef Eigen::SparseMatrix<SReal, Eigen::RowMajor> SMatrix;
    typedef linearsolver::EigenVector<SReal>            VectorSofa;
    typedef Eigen::Matrix<SReal, Eigen::Dynamic, 1>     VectorEigen;
    typedef core::behavior::BaseMechanicalState         MechanicalState;
    typedef core::BaseMapping                           Mapping;

private:
    // Equation system
    SMatrix _matM;         ///< Mass matrix, also used to store the implicit matrix M/h - Kh
    SMatrix _matK;         ///< Stiffness matrix
    SMatrix _matJ;         ///< concatenation of the constraint Jacobians, in the KKT system
    SMatrix _matC;         ///< compliance matrix used to regularize the system, in the bottom-right of the KKT system
    VectorEigen _vecF;      ///< top of the right-hand term: forces
    VectorEigen _vecPhi;    ///< bottom of the right-hand term: desired constraint corrections
    VectorEigen _vecDv;     ///< top of the solution: velocity change
    VectorEigen _vecLambda; ///< bottom of the solution: Lagrange multipliers
    linearsolver::EigenBaseSparseMatrix<SReal> _projMatrix;
    linearsolver::EigenBaseSparseMatrix<SReal> _PMinvP_Matrix;  ///< PMinvP

    bool _PMinvP_isDirty;  ///< true if _PMinvP_Matrix is not up to date
    VectorEigen _vecV;      ///< velocties, used in the computation of the right-hand term of the implicit equation

public:
    // Equation system
    /// Top-left block of the equation matrix: mass, possibly combined with stiffness and damping
    const SMatrix& M() const { return _matM; }
    /// Jacobian of the (rigid or compliant) constraints, bottom-left block of the equation matrix, and opposite of the top-right block
    const SMatrix& J() const { return _matJ; }
    /// Bottom-right of the equation matrix. Compliance of the constraints, null for hard constraints.
    const SMatrix& C() const { return _matC; }
    /// Projection matrix to implement simple imposed displacements
    const SMatrix& P() const { return _projMatrix.compressedMatrix; }
    /// Projected M
    const SMatrix& PMinvP();

    /// Upper part of the right-hand term of the dynamics equation
    const VectorEigen& f() const { return _vecF; }
    /// Lower part of the right-hand term of the dynamics equation
    const VectorEigen& phi() const { return _vecPhi; }
    /// Upper part of the unknown of the dynamics equation
    const VectorEigen& dv() const { return _vecDv; }
    /// Lower part of the unknown of the dynamics equation
    const VectorEigen& lambda() const { return _vecLambda; }

    // this redundant interface is required for the test fixture because strangely enough, it sees dv() and lambda() as protected instead of public, even for read access
    /// Upper part of the unknown of the dynamics equation
    const VectorEigen& getDv() const { return _vecDv; }
    /// Lower part of the unknown of the dynamics equation
    const VectorEigen& getLambda() const { return _vecLambda; }

protected:
    /// Upper part of the unknown of the dynamics equation: write access for the derived classes
    VectorEigen& dv() { return _vecDv; }
    /// Lower part of the unknown of the dynamics equation: write access for the derived classes
    VectorEigen& lambda() { return _vecLambda; }

protected:
    /** Solve the equation system:

    \f$
    \left( \begin{array}{cc} PM & -PJ^T \\
                               J &  C \end{array}\right)
      \left( \begin{array}{c} \Delta v \\ \lambda \end{array}\right)
    = \left( \begin{array}{c} Pf \\  \phi  \end{array}\right)
    \f$

    where the values of matrices \f$M,P,J,C\f$ and vectors \f$f, \phi\f$ depend on the integration scheme (see the class documentation), and are stored in members  matM, matP, matJ, matC, vecF and vecPhi, respectively.
    Additionally, matrix \f$  PM^{-1}P \f$ is available as member PMinvP.
    The solution \f$ \Delta v, \lambda \f$ computed by the method is stored in members vecDv and vecLambda, respectively.

     The base implementation uses a direct Cholesky solver. It can be overloaded to apply other linear equation solvers.
     */
    virtual void solveEquation();

public:
    Data<SReal> implicitVelocity;     ///< the \f$ \alpha \f$ parameter of the integration scheme
    Data<SReal> implicitPosition;     ///< the \f$ \beta  \f$ parameter of the integration scheme
    Data<SReal> f_rayleighStiffness;  ///< uniform Rayleigh damping ratio applied to the stiffness matrix
    Data<SReal> f_rayleighMass;       ///< uniform Rayleigh damping ratio applied to the mass matrix
    Data<bool>  verbose;              ///< print a lot of debug info

protected:


    typedef enum { COMPUTE_SIZE, DO_SYSTEM_ASSEMBLY, /*PROJECT_MATRICES,*/ DISTRIBUTE_SOLUTION } Pass;  ///< Symbols of operations to execute by the visitor

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

        /// Casts the matrix using a dynamic_cast. Crashes if the BaseMatrix* is not a SMatrix*
        static const SMatrix& getSMatrix( const defaulttype::BaseMatrix* );
    };

    // sparse LDLT support (requires  SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD compile flag)
    typedef Eigen::SparseLDLT<Eigen::SparseMatrix<SReal>,Eigen::Cholmod>  SparseLDLT;  // process SparseMatrix, not DynamicSparseMatrix (not implemented in Cholmod)

    /// Return a rectangular matrix (cols>rows), with (offset-1) null columns, then the (rows*rows) identity, then null columns.
    /// This is used to shift a "local" matrix to the global indices of an assembly matrix.
    static SMatrix createShiftMatrix( unsigned rows, unsigned cols, unsigned offset );

    /// Compute the inverse of the matrix. The input matrix MUST be diagonal. Return false if the matrix is not diagonal. The threshold parameter is currently unused.
    static bool inverseDiagonalMatrix( SMatrix& minv, const SMatrix& m );

    /// Compute the inverse of the matrix.
    static void inverseMatrix( SMatrix& minv, const SMatrix& m );

    /// Return an identity matrix of the given size
    static SMatrix createIdentityMatrix( unsigned size );


    /** Local matrices and offsets associated with a given MechanicalState, and their offsets in the assembled independent DOFs.
      This can be used either to perform final assembly by summing products, e.g. _matM += J.transpose() * M * J
      or to compute global matrix-vector products by looping over all the entries and accumulating products, e.g. df += J.transpose() * (K * (J * dx))
      */
    struct LocalMatrices
    {
        SMatrix M;         ///< local mass matrix
        SMatrix J;         ///< local Jacobian wrt assembled independent DOFs:   v_local = J * v_global
        SMatrix C;         ///< local compliance matrix, if any
        SMatrix K;         ///< local stiffness matrix, if any
        unsigned m_offset; ///< start index in the assembled mass matrix
        unsigned c_offset; ///< start index in the assembled compliance matrix

        friend std::ostream& operator << (std::ostream& out, const LocalMatrices& sm )
        {
            out << "m_offset = " << sm.m_offset << ", c_offset = "<< sm.c_offset << endl;
            if(sm.M.rows()>0) out << "M=" << endl << DenseMatrix(sm.M) << endl;
            if(sm.J.rows()>0) out << "J=" << endl << DenseMatrix(sm.J) << endl;
            if(sm.C.rows()>0) out << "C=" << endl << DenseMatrix(sm.C) << endl;
            if(sm.K.rows()>0) out << "K=" << endl << DenseMatrix(sm.K) << endl;
            return out;
        }
    };
    typedef std::map<core::behavior::BaseMechanicalState*,LocalMatrices> State_2_LocalMatrices;
    State_2_LocalMatrices localMatrices;         ///< The local matrices associated to each state
    void writeLocalMatrices() const;             ///< debug helper


    /** @name Global matrix-vector product */
    ///@{

    /// Compute the product of a vector with implicit matrix M, which may not be assembled. M may be the implicit matrix e.g. M-h²K
    struct MatrixProduct
    {
    protected:
        ComplianceSolver& solver;
        mutable VectorEigen product;
    public:

        MatrixProduct( ComplianceSolver& s ) : solver(s) {}

        /// Product of a vector with matrix M. The default implementation assumes that M is assembled.
        virtual const VectorEigen& operator()(const VectorEigen& x) const
        {
            product.noalias() = solver.M() * x;
            return product;
        }
    };


    /// Compute the product of a vector with matrix M, in case the J products are computed but matrix M is not assembled
    struct JtAJProduct: public MatrixProduct
    {
        JtAJProduct( ComplianceSolver& s ) : MatrixProduct(s) {}

        /// Product of a vector with matrix M.
        virtual const VectorEigen& operator()(const VectorEigen& x) const
        {
            product.resize(x.size());
            solver.multM(product,x);
            return product;
        }
    };



    /// Compute the product of the non-assembled implicit matrix with a vector, using local M as implicit matrix
    void multM(VectorEigen& Mx, const VectorEigen& x)
    {
        for( State_2_LocalMatrices::iterator i=localMatrices.begin(), iend=localMatrices.end(); i!=iend; i++ )
        {
            const SMatrix& J = (*i).second.J;
            const SMatrix& M = (*i).second.M;
            Mx += J.transpose() * (M * (J * x));
        }
    }
    ///@}  // end group matrix-vector product



};

}
}
}

#endif // SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
