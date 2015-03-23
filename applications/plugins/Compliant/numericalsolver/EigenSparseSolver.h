#ifndef COMPLIANT_EIGENSPARSESOLVER_H
#define COMPLIANT_EIGENSPARSESOLVER_H

#include "KKTSolver.h"
#include "Response.h"

#include "../utils/scoped.h"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace sofa {
namespace component {
namespace linearsolver {

/// Solve a dynamics system including bilateral constraints
///
/// schur = 1
/// Schur complement solve
/// Note that the dynamics equation is solved by an external Response component
///
/// schur = 0
/// KKT system solve
///
template<class LinearSolver, bool symmetric=false, int UpLo=0>
class EigenSparseSolver : public KKTSolver {
  public:

    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE3(EigenSparseSolver,LinearSolver,symmetric,UpLo), KKTSolver);

    virtual void solve(vec& x,
                       const AssembledSystem& system,
                       const vec& rhs) const;

    // performs factorization
    virtual void factor(const AssembledSystem& system);

    virtual void init();

    EigenSparseSolver();
    ~EigenSparseSolver();

  protected:

    // response matrix
    Response::SPtr response;

    void factor_schur(const AssembledSystem& system);

    void solve_schur(vec& x, const AssembledSystem& system, const vec& rhs) const;
    void solve_kkt(vec& x, const AssembledSystem& system, const vec& rhs) const;

    Data<bool> schur;

    struct pimpl_type;
    scoped::ptr<pimpl_type> pimpl;


};



typedef Eigen::SimplicialLDLT< AssembledSystem::cmat > LDLTSparseLinearSolver;
class SOFA_Compliant_API LDLTSolver : public EigenSparseSolver< LDLTSparseLinearSolver, true, LDLTSparseLinearSolver::UpLo >
{
public:
    SOFA_CLASS(LDLTSolver,SOFA_TEMPLATE3(EigenSparseSolver,LDLTSparseLinearSolver,true,LDLTSparseLinearSolver::UpLo));
};


typedef Eigen::SparseLU< AssembledSystem::cmat > LUSparseLinearSolver;
class SOFA_Compliant_API LUSolver : public EigenSparseSolver< LUSparseLinearSolver >
{
public:
    SOFA_CLASS(LUSolver,SOFA_TEMPLATE(EigenSparseSolver,LUSparseLinearSolver));
};






/////////////////////////////////////////////


/// Solve a dynamics system including bilateral constraints with an iterative linear solver
template<class LinearSolver, bool symmetric=false, int UpLo=0>
class EigenSparseIterativeSolver : public EigenSparseSolver<LinearSolver,symmetric,UpLo>
{
  public:

    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE3(EigenSparseIterativeSolver,LinearSolver,symmetric,UpLo),SOFA_TEMPLATE3(EigenSparseSolver,LinearSolver,symmetric,UpLo));

    Data<unsigned> d_iterations;
    Data<SReal> d_tolerance;

    EigenSparseIterativeSolver();

    virtual void init();
    virtual void reinit();

};



// TODO check why passing only a triangular matrix to the symmetric iterative solvers does not work


typedef Eigen::ConjugateGradient< AssembledSystem::cmat > CGSparseLinearSolver;
class EigenCGSolver : public EigenSparseIterativeSolver< CGSparseLinearSolver, true >
{
public:
    SOFA_CLASS(EigenCGSolver,SOFA_TEMPLATE2(EigenSparseIterativeSolver,CGSparseLinearSolver,true));
};


typedef Eigen::BiCGSTAB< AssembledSystem::cmat > BiCGSTABSparseLinearSolver;
class EigenBiCGSTABSolver : public EigenSparseIterativeSolver< BiCGSTABSparseLinearSolver >
{
public:
    SOFA_CLASS(EigenBiCGSTABSolver,SOFA_TEMPLATE(EigenSparseIterativeSolver,BiCGSTABSparseLinearSolver));
};


typedef Eigen::MINRES< AssembledSystem::cmat > MINRESSparseLinearSolver;
class EigenMinresSolver : public EigenSparseIterativeSolver< MINRESSparseLinearSolver, true >
{
public:
    SOFA_CLASS(EigenMinresSolver,SOFA_TEMPLATE2(EigenSparseIterativeSolver,MINRESSparseLinearSolver,true));
};

}
}
}



#endif

