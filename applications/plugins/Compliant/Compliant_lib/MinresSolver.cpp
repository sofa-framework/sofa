#include "MinresSolver.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>
//#include <sofa/component/linearsolver/EigenSparseSquareMatrix.h>
#include <sofa/component/linearsolver/SingleMatrixAccessor.h>

#include <iostream>

#include "utils/minres.h"
#include "utils/cg.h"

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
int MinresSolverClass =
    core::RegisterObject("Variant of ComplianceSolver where a minres iterative solver is used in place of the direct solver")
    .add< MinresSolver >();

// scoped logging
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




// schur system functor
struct MinresSolver::schur
{

    const mat& Minv;
    const mat& J;
    const mat& C;

    mutable vec storage, Jx, MinvJx, Cx, JMinvJx;

    schur(const mat& Minv,
            const mat& J,
            const mat& C)
        : Minv(Minv),
          J(J),
          C(C),
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

        storage.noalias() = JMinvJx + Cx;
        return storage;
    }

};



void MinresSolver::solve_schur(krylov::params& p )
{
    raii_log log("MinresSolver::solve_schur");

    const vec b = phi() - J() * ( PMinvP() * f() );

    vec x = vec::Zero( b.size() );
    warm( x );

    // schur matrix
    schur A(PMinvP(), J(), C());

    if(use_cg.getValue())
    {
        cg::solve(x, A, b, p);
    }
    else
    {
        minres::solve(x, A, b, p);
    }

    last = x;

    lambda() = x;

    const vec ftmp = f() + J().transpose() * lambda();
    dv() = PMinvP() * ftmp;
    // the following is MUCH slower:
//        dv() = PMinvP() * (f() + J().transpose() * lambda());
}

void MinresSolver::warm(vec& x) const
{
    if( use_warm.getValue() && (x.size() == last.size()) )
    {
        x = last;
    }
}




// kkt system functor
struct MinresSolver::kkt
{

    const mat& M;
    const mat& J;
    const mat& P;
    const mat& C;

    const int m, n;

    mutable vec storage, Mx, Px, JPx, JTlambda, PTJTlambda, Clambda;

    kkt(const mat& M,
        const mat& J,
        const mat& P,
        const mat& C)
        : M(M),
          J(J),
          P(P),
          C(C),
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
                PTJTlambda.noalias() = P.transpose() * JTlambda; // TODO is this optimized ? is P symmetric ?
            }
            #pragma omp section
            Clambda.noalias() = C * x.tail(n);
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            storage.head(m).noalias() = Mx - PTJTlambda;

            #pragma omp section
            storage.tail(n).noalias() = -JPx - Clambda;
        }

        return storage;
    }

};






void MinresSolver::solve_kkt(krylov::params& p )
{
    raii_log log("MinresSolver::solve_kkt");
    vec b; b.resize(f().size() + phi().size());

    // TODO f projection is probably not needed
    b << P() * f(), -phi();

    vec x = vec::Zero( b.size() );
    warm(x);

    // kkt matrix
    kkt A(M(), J(), P(), C());

    if(use_cg.getValue())
    {
        // cg::solve(x, A, b, p);
    }
    else
    {
        minres::solve(x, A, b, p);
    }

    last = x;

    dv() = P() * x.head( f().size() );
    lambda() = x.tail( phi().size() );

}


void MinresSolver::solveEquation()
{
    // setup minres
    krylov::params p;
    p.iterations = max_iterations.getValue();
    p.precision = precision.getValue();

    // solve for lambdas
//        lambda() = use_kkt.getValue() ? solve_kkt(p) : solve_schur(p);
    if(use_kkt.getValue())
    {
        solve_kkt(p);
    }
    else
    {
        solve_schur(p);
    }

    iterations_performed.setValue( p.iterations );

//	// now done within the equation solver
//        this->f() += J().transpose() * lambda();
//        dv() = PMinvP() * f();  // (FF)
}


MinresSolver::MinresSolver()
    : use_kkt( initData(&use_kkt, false, "kkt",
            "Work on KKT system instead of Schur complement ?") ),

    max_iterations( initData(&max_iterations, (unsigned int)(100), "maxIterations",
            "Solver iterations bound")),

    iterations_performed( initData(&iterations_performed, (unsigned int)(0), "iterationsPerformed",
            "Iterations performed during the last time step (read-only)")),

    precision( initData(&precision, 1e-7, "precision",
            "Residual threshold")),

    use_warm( initData(&use_warm, false, "warm",
            "Warm start solver ?")),

    use_cg( initData(&use_cg, false, "cg",
            "Use CG instead of MINRES ?"))
{

}
}
}
}
