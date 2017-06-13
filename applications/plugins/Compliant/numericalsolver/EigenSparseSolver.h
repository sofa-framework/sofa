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
/// @author Matthieu Nesme
///
template<class LinearSolver, bool symmetric=false>
class EigenSparseSolver : public KKTSolver {
  public:

    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(EigenSparseSolver,LinearSolver,symmetric), KKTSolver);

    virtual void solve(vec& x,
                       const AssembledSystem& system,
                       const vec& rhs) const;

    // performs factorization
    virtual void factor(const AssembledSystem& system);

    virtual void init();
    virtual void reinit();

    EigenSparseSolver();
    ~EigenSparseSolver();


    Data<bool> d_schur;
    Data<SReal> d_regularization; ///< Optional diagonal Tikhonov regularization on constraints
    /// if the sparsity pattern remains similar from one step to another,
    /// the factorization can be faster
    Data<bool> d_trackSparsityPattern;


  protected:

    // response matrix
    Response::SPtr response;

    void factor_schur(const AssembledSystem& system);

    void solve_schur(vec& x, const AssembledSystem& system, const vec& rhs) const;
    void solve_kkt(vec& x, const AssembledSystem& system, const vec& rhs) const;

    struct pimpl_type;
    scoped::ptr<pimpl_type> pimpl;

};



typedef Eigen::SimplicialLDLT< AssembledSystem::cmat > LDLTSparseLinearSolver;
class SOFA_Compliant_API LDLTSolver : public EigenSparseSolver< LDLTSparseLinearSolver, true >
{
public:
    SOFA_CLASS(LDLTSolver,SOFA_TEMPLATE2(EigenSparseSolver,LDLTSparseLinearSolver,true));
};

typedef Eigen::SimplicialLLT< AssembledSystem::cmat > LLTSparseLinearSolver;
class SOFA_Compliant_API LLTSolver : public EigenSparseSolver< LLTSparseLinearSolver, true >
{
public:
    SOFA_CLASS(LLTSolver,SOFA_TEMPLATE2(EigenSparseSolver,LLTSparseLinearSolver,true));
};

typedef Eigen::SparseLU< AssembledSystem::cmat > LUSparseLinearSolver;
class SOFA_Compliant_API LUSolver : public EigenSparseSolver< LUSparseLinearSolver >
{
public:
    SOFA_CLASS(LUSolver,SOFA_TEMPLATE(EigenSparseSolver,LUSparseLinearSolver));
};




/////////////////////////////////////////////


/// Solve a dynamics system including bilateral constraints with an iterative linear solver
template<class LinearSolver, bool symmetric=false>
class EigenSparseIterativeSolver : public EigenSparseSolver<LinearSolver,symmetric>
{
public:

    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(EigenSparseIterativeSolver,LinearSolver,symmetric),SOFA_TEMPLATE2(EigenSparseSolver,LinearSolver,symmetric));

    Data<unsigned> d_iterations; ///< max iterations
    Data<SReal> d_tolerance; ///< tolerance

    EigenSparseIterativeSolver();

    virtual void init();
    virtual void reinit();

};



typedef Eigen::ConjugateGradient< AssembledSystem::cmat > CGSparseLinearSolver;
class EigenCGSolver : public EigenSparseIterativeSolver< CGSparseLinearSolver, true >
{
public:
    SOFA_CLASS(EigenCGSolver,SOFA_TEMPLATE2(EigenSparseIterativeSolver,CGSparseLinearSolver, true));
};


typedef Eigen::BiCGSTAB< AssembledSystem::cmat > BiCGSTABSparseLinearSolver;
class EigenBiCGSTABSolver : public EigenSparseIterativeSolver< BiCGSTABSparseLinearSolver >
{
public:
    SOFA_CLASS(EigenBiCGSTABSolver,SOFA_TEMPLATE(EigenSparseIterativeSolver,BiCGSTABSparseLinearSolver));
};


typedef Eigen::MINRES< AssembledSystem::cmat > MINRESSparseLinearSolver;
class EigenMINRESSolver : public EigenSparseIterativeSolver< MINRESSparseLinearSolver, true >
{
public:
    SOFA_CLASS(EigenMINRESSolver,SOFA_TEMPLATE2(EigenSparseIterativeSolver,MINRESSparseLinearSolver, true));
};

typedef Eigen::GMRES< AssembledSystem::cmat > GMRESSparseLinearSolver;
class EigenGMRESSolver : public EigenSparseIterativeSolver< GMRESSparseLinearSolver >
{
public:
    SOFA_CLASS(EigenGMRESSolver,SOFA_TEMPLATE(EigenSparseIterativeSolver,GMRESSparseLinearSolver));
};




}
}
}



#endif

