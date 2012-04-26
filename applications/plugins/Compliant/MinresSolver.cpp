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
            storage.tail(n).noalias() = -JPx - Clambda;
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

    mutable vec storage, Jx, MinvJx, Cx, JMinvJx;

    schur(const SMatrix& Minv,
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


//      const MinresSolver::mat& MinresSolver::M() const {
//	return _matM;
//      }

//      MinresSolver::mat& MinresSolver::M() {
//	return _matM;
//      }


//      const MinresSolver::mat& MinresSolver::J() const {
//	return _matJ;
//      }

//      MinresSolver::mat& MinresSolver::J() {
//	return _matJ;
//      }

//      const MinresSolver::mat& MinresSolver::C() const {
//	return _matC;
//      }

//      MinresSolver::mat& MinresSolver::C() {
//	return _matC;
//      }


//      const MinresSolver::mat& MinresSolver::projMatrix() const {
//          return P();
//      }

//      MinresSolver::mat& MinresSolver::projMatrix() {
//	return _matP;
//      }


//      MinresSolver::vec& MinresSolver::f() {
//	return _vecF.getVectorEigen();
//      }

//      const MinresSolver::vec& MinresSolver::f() const {
//        return vecF().getVectorEigen();
//      }


//      const MinresSolver::vec& MinresSolver::phi() const {
//	return _vecPhi.getVectorEigen();
//      }

//      MinresSolver::vec& MinresSolver::phi() {
//        return vecPhi().getVectorEigen();
//      }


void MinresSolver::solve_schur(minres::params& p )
{
    raii_log log("MinresSolver::solve_schur");


    const vec rhs = phi() - J() * ( PMinvP() * f() );

    vec mylambda = vec::Zero( rhs.size() );
    warm(mylambda);

    ::minres<double>::solve<schur>(mylambda, schur(PMinvP(), J(), C()), rhs, p);

    last = mylambda;

    lambda() = mylambda;


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

void MinresSolver::solve_kkt(minres::params& p )
{
    raii_log log("MinresSolver::solve_kkt");
    vec rhs; rhs.resize(f().size() + phi().size());

    // TODO f projection is probably not needed
    rhs << P() * f(), -phi();

    vec x = vec::Zero( rhs.size() );
    warm(x);

    minres::solve(x, kkt(M(), J(), P(), C() ), rhs, p);
    last = x;

    dv() = P() * x.head( f().size() );
    lambda() = x.tail( phi().size() );

}


void MinresSolver::solveEquation()
{
    // setup minres
    minres::params p;
    p.iterations = max_iterations.getValue();
    p.precision = precision.getValue();

    // solve for lambdas
//        lambda() = use_kkt.getValue() ? solve_kkt(p) : solve_schur(p);
    if(use_kkt.getValue())
        solve_kkt(p);
    else
        solve_schur(p);

    iterations_performed.setValue( p.iterations );

//	// now done within the equation solver
//        this->f() += J().transpose() * lambda();
//        dv() = PMinvP() * f();  // (FF)
}


MinresSolver::MinresSolver()
    : use_kkt( initData(&use_kkt, false, "kkt",
            "Work on KKT system instead of Schur complement") ),

    max_iterations( initData(&max_iterations, (unsigned int)(100), "maxIterations",
            "Iterations bound for the MINRES solver")),

    iterations_performed( initData(&iterations_performed, (unsigned int)(0), "iterationsPerformed",
            "Iterations actually performed by the MINRES solver for the last time step")),

    precision( initData(&precision, 1e-7, "precision",
            "Residual threshold for the MINRES solver")),

    use_warm( initData(&use_warm, false, "warm",
            "Warm start MINRES"))
{

}
}
}
}
