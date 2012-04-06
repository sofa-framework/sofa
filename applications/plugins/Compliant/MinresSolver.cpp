#include "MinresSolver.h"

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


struct raii_log
{
    const std::string name;

    raii_log(const std::string& name) : name(name)
    {
        simulation::Visitor::printNode(name);
    }

    ~raii_log()
    {
        simulation::Visitor::printCloseNode(name);
    }

};


// kkt system functor
struct MinresSolver::kkt
{

    const mat& M;
    const mat& J;
    const mat& P;
    const mat& C;

    const double dt;
    const int m, n;

    mutable vec storage, Mx, Px, JPx, JTlambda, PTJTlambda, Clambda;

    kkt(const mat& M,
        const mat& J,
        const mat& P,
        const mat& C,
        double dt)
        : M(M),
          J(J),
          P(P),
          C(C),
          dt(dt),
          m( M.rows() ),
          n( J.rows() ),
          storage( vec::Zero(m + n ) )
    {

    }

    const vec& operator()(const vec& x) const
    {

        // let's avoid allocs and use omp
        #pragma omp parallel sections
        {
            #pragma omp section
            Mx.noalias() = M * x.head(m);

            #pragma omp section
            {
                Px.noalias() = P * x.head(m);
                JPx.noalias() = J * Px;
            }

            #pragma omp section
            {
                JTlambda.noalias() = J.transpose() * x.tail(n);
                PTJTlambda.noalias() = P * JTlambda; // should be P.transpose()
            }
            #pragma omp section
            Clambda.noalias() = C * x.tail(n);
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            storage.head(m).noalias() = Mx - PTJTlambda;

            #pragma omp section
            storage.tail(n).noalias() = -JPx - Clambda; // should be / (dt * dt)
        }

        return storage;
    }

};



// schur system functor
struct MinresSolver::schur
{

    const SMatrix& Minv;
    const mat& J;
    const mat& C;

    const double dt;

    mutable vec storage, Jx, MinvJx, Cx, JMinvJx;

    schur(const SMatrix& Minv,
            const mat& J,
            const mat& C,
            double dt)
        : Minv(Minv),
          J(J),
          C(C),
          dt(dt),
          storage( vec::Zero( J.rows() ) )
    {

    }

    const vec& operator()(const vec& x) const
    {

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                Jx.noalias() = J.transpose() * x;
                MinvJx.noalias() = Minv * Jx;
                JMinvJx.noalias() = J * MinvJx;
            }
            #pragma omp section
            Cx.noalias() = C * x;
        }

        storage.noalias() = JMinvJx + Cx; // should be: Cx / (dt * dt);
        return storage;
    }

};


void MinresSolver::bwdInit()
{
    core::ExecParams params;
}


const MinresSolver::mat& MinresSolver::M() const
{
    return matM;
}

MinresSolver::mat& MinresSolver::M()
{
    return matM;
}


const MinresSolver::mat& MinresSolver::J() const
{
    return matJ;
}

MinresSolver::mat& MinresSolver::J()
{
    return matJ;
}

const MinresSolver::mat& MinresSolver::C() const
{
    return matC;
}

MinresSolver::mat& MinresSolver::C()
{
    return matC;
}


const MinresSolver::mat& MinresSolver::P() const
{
    return matP;
}

MinresSolver::mat& MinresSolver::P()
{
    return matP;
}


MinresSolver::vec& MinresSolver::f()
{
    return vecF.getVectorEigen();
}

const MinresSolver::vec& MinresSolver::f() const
{
    return vecF.getVectorEigen();
}


const MinresSolver::vec& MinresSolver::phi() const
{
    return vecPhi.getVectorEigen();
}

MinresSolver::vec& MinresSolver::phi()
{
    return vecPhi.getVectorEigen();
}

MinresSolver::visitor::visitor(core::MechanicalParams* cparams, MinresSolver* s)
    : assembly(cparams, s),
      solver(s)
{

}


MinresSolver::visitor MinresSolver::make_visitor(core::MechanicalParams* cparams)
{
    return visitor(cparams, this);
}

bool MinresSolver::visitor::fetch()
{
    raii_log log("assembly_visitor");

    solver->getContext()->executeVisitor(&assembly(COMPUTE_SIZE));
    //    cerr<<"ComplianceSolver::solve, sizeM = " << assembly.sizeM <<", sizeC = "<< assembly.sizeC << endl;

    if( assembly.sizeC > 0 )
    {

        solver->M().resize(assembly.sizeM,assembly.sizeM);
        solver->P().resize(assembly.sizeM,assembly.sizeM);
        solver->C().resize(assembly.sizeC,assembly.sizeC);
        solver->J().resize(assembly.sizeC,assembly.sizeM);
        solver->f().resize(assembly.sizeM);
        solver->phi().resize(assembly.sizeC);
        solver->vecF.clear();
        solver->vecPhi.clear();

        // Matrix assembly
        solver->getContext()->executeVisitor(&assembly(DO_SYSTEM_ASSEMBLY));

        // solver->matC.setZero();

        // if( solver->verbose.getValue() )
        //   {
        //     cerr<<"ComplianceSolver::solve, final M = " << endl << matM << endl;
        //     cerr<<"ComplianceSolver::solve, final P = " << endl << matP << endl;
        //     cerr<<"ComplianceSolver::solve, final C = " << endl << matC << endl;
        //     cerr<<"ComplianceSolver::solve, final J = " << endl << matJ << endl;
        //     cerr<<"ComplianceSolver::solve, final f = " << vecF << endl;
        //     cerr<<"ComplianceSolver::solve, final phi = " << vecPhi << endl;
        //   }
        return true;
    }
    else
    {
        return false;
    }

}


void MinresSolver::visitor::distribute()
{
    solver->getContext()->executeVisitor(&assembly(DISTRIBUTE_SOLUTION));  // set dv in each MechanicalState
}


MinresSolver::vec MinresSolver::solve_schur(real dt, const minres::params& p )  const
{
    raii_log log("minres_schur");
    SMatrix Minv = inverseMatrix( M(), 1.0e-6 );
    Minv = matP * Minv * matP;

    const vec rhs = phi() - J() * ( Minv * this->f() );

    vec lambda = vec::Zero( rhs.size() );
    warm(lambda);

    minres::solve(lambda, schur(Minv, J(), C(), dt), rhs, p);
    last = lambda;

    return lambda;
}

void MinresSolver::warm(vec& x) const
{
    if( use_warm.getValue() && (x.size() == last.size()) )
    {
        x = last;
    }
}

MinresSolver::vec MinresSolver::solve_kkt( real dt, const minres::params& p ) const
{
    raii_log log("minres_kkt");
    vec rhs; rhs.resize(f().size() + phi().size());

    // TODO f projection is probably not needed
    rhs << P() * f(), -phi();

    vec x = vec::Zero( rhs.size() );
    warm(x);

    minres::solve(x, kkt(M(), J(), P(), C(), dt), rhs, p);
    last = x;

    return x.tail( phi().size() );
}


void MinresSolver::solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    // tune parameters
    core::MechanicalParams cparams(*params);
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

    // create assembly visitor
    visitor vis = make_visitor( &cparams );

    // if there is something to solve
    if( vis.fetch() )
    {

        // setup minres
        minres::params p;
        p.iterations = iterations.getValue();
        p.precision = precision.getValue();

        // solve for lambdas
        vec lambda = use_kkt.getValue() ? solve_kkt(dt, p) : solve_schur(dt,  p);

        // add constraint force
        this->f() += J().transpose() * lambda;

        // dispatch constraint forces
        vis.distribute();
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


MinresSolver::MinresSolver()
    : use_kkt( initData(&use_kkt, false, "kkt",
            "Work on KKT system instead of Schur complement") ),

    iterations( initData(&iterations, (unsigned int)(100), "iterations",
            "Iterations bound for the MINRES solver")),

    precision( initData(&precision, 1e-7, "precision",
            "Residual threshold for the MINRES solver")),

    use_warm( initData(&use_warm, false, "warm",
            "Warm start MINRES"))
{

}
}
}
}
