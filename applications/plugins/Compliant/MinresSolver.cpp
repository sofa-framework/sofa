#include "MinresSolver.h"
#include "BaseCompliance.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>
#include <sofa/component/linearsolver/EigenSparseSquareMatrix.h>
#include <sofa/component/linearsolver/SingleMatrixAccessor.h>

#include <iostream>

#include "utils/minres.h"


using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;

SOFA_DECL_CLASS(MinresSolver);
int MinresSolverClass = core::RegisterObject("A simple explicit time integrator").add< MinresSolver >();


typedef Eigen::VectorXd vec;
typedef Eigen::DynamicSparseMatrix<double, Eigen::RowMajor> mat;

struct kkt
{

    const mat& M, J, C;
    const double dt;
    const int m, n;

    mutable vec storage;

    kkt(const mat& M,
        const mat& J,
        const mat& C,
        double dt)
        : M(M),
          J(J),
          C(C),
          dt(dt),
          m( M.rows() ),
          n( J.rows() ),
          storage( vec::Zero(m + n ) )
    {

    }

    const vec& operator()(const vec& x) const
    {

        storage.head(m) = M * x.head(m) + J.transpose() * x.tail(n);
        storage.tail(n) = J * x.head(m) - ( C * x.tail(n) ) / (dt * dt);

        return storage;
    }

};


void MinresSolver::solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    // tune parameters
    core::ComplianceParams cparams(*params);
    cparams.setMFactor(1.0);
    cparams.setDt(dt);
    cparams.setImplicitVelocity( implicitVelocity.getValue() );
    cparams.setImplicitPosition( implicitPosition.getValue() );

    //  State vectors and operations
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );

    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecDeriv dv(&vop, core::VecDerivId::dx() );
    MultiVecDeriv f  (&vop, core::VecDerivId::force() );
    MultiVecCoord nextPos(&vop, xResult );
    MultiVecDeriv nextVel(&vop, vResult );


    // Compute right-hand term
    mop.computeForce(f);
    mop.projectResponse(f);
    if( verbose.getValue() )
    {
        cerr<<"ComplianceSolver::solve, filtered external forces = " << f << endl;
    }

    // Matrix size
    MatrixAssemblyVisitor assembly(&cparams,this);
    this->getContext()->executeVisitor(&assembly(COMPUTE_SIZE));
    //    cerr<<"ComplianceSolver::solve, sizeM = " << assembly.sizeM <<", sizeC = "<< assembly.sizeC << endl;

    if( assembly.sizeC > 0 )
    {

        matM.resize(assembly.sizeM,assembly.sizeM);
        matP.resize(assembly.sizeM,assembly.sizeM);
        matC.resize(assembly.sizeC,assembly.sizeC);
        matJ.resize(assembly.sizeC,assembly.sizeM);
        vecF.resize(assembly.sizeM);
        vecPhi.resize(assembly.sizeC);

        // Matrix assembly
        this->getContext()->executeVisitor(&assembly(MATRIX_ASSEMBLY));

        // Vector assembly  (do we need a separate pass ?)
        vecF.clear();
        vecPhi.clear();
        this->getContext()->executeVisitor(&assembly(VECTOR_ASSEMBLY));

        if( verbose.getValue() )
        {
            cerr<<"ComplianceSolver::solve, final M = " << endl << matM << endl;
            cerr<<"ComplianceSolver::solve, final P = " << endl << matP << endl;
            cerr<<"ComplianceSolver::solve, final C = " << endl << matC << endl;
            cerr<<"ComplianceSolver::solve, final J = " << endl << matJ << endl;
            cerr<<"ComplianceSolver::solve, final f = " << vecF << endl;
            cerr<<"ComplianceSolver::solve, final phi = " << vecPhi << endl;
        }

        const int m = assembly.sizeM;
        const int n = assembly.sizeC;

        vec rhs = vec::Zero( m + n );

        rhs.head(m) = vecF.getVectorEigen();
        rhs.tail(m) = vecPhi.getVectorEigen();

        typedef minres<double> solver;
        solver::params p;

        p.iterations = 100;
        p.precision = 1e-7;

        vec x = vec::Zero(m + n);

        solver::solve(x, kkt(matM, matJ, matC, dt), rhs, p);
        vecF.getVectorEigen() = vecF.getVectorEigen() + matJ.transpose() * x.tail(n);

        this->getContext()->executeVisitor(&assembly(VECTOR_DISTRIBUTE));  // set dv in each MechanicalState
    }

    mop.accFromF(dv, f);
    mop.projectResponse(dv);


    // Apply integration scheme

    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp ops;
    ops.resize(2);
    ops[0].first = nextPos; // p = p + v*h + dv*h*beta
    ops[0].second.push_back(std::make_pair(pos.id(),1.0));
    ops[0].second.push_back(std::make_pair(vel.id(),dt));
    ops[0].second.push_back(std::make_pair(  dv.id(),dt*implicitPosition.getValue()));
    ops[1].first = nextVel; // v = v + dv
    ops[1].second.push_back(std::make_pair(vel.id(),1.0));
    ops[1].second.push_back(std::make_pair(  dv.id(),1.0));
    vop.v_multiop(ops);

    if( verbose.getValue() )
    {
        mop.propagateX(nextPos);
        mop.propagateDx(nextVel);
        serr<<"EulerImplicitSolver, final x = "<< nextPos <<sendl;
        serr<<"EulerImplicitSolver, final v = "<< nextVel <<sendl;
    }

}



}
}
}
