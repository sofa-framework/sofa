#ifndef COMPLIANT_EIGENSPARSESOLVER_H
#define COMPLIANT_EIGENSPARSESOLVER_H

#include "KKTSolver.h"
#include "Response.h"

#include "../utils/scoped.h"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>


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
template<class LinearSolver, bool symmetric>
class SOFA_Compliant_API EigenSparseSolver : public KKTSolver {
  public:

    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(EigenSparseSolver,LinearSolver,symmetric), KKTSolver);

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

  private:

    struct pimpl_type;
    scoped::ptr<pimpl_type> pimpl;


};



class SOFA_Compliant_API LDLTSolver : public EigenSparseSolver< Eigen::SimplicialLDLT< AssembledSystem::cmat >, true >
{
public:
    SOFA_CLASS(LDLTSolver,SOFA_TEMPLATE2(EigenSparseSolver,SOFA_TEMPLATE(Eigen::SimplicialLDLT,AssembledSystem::cmat),true));
};


class SOFA_Compliant_API LUSolver : public EigenSparseSolver< Eigen::SparseLU< AssembledSystem::cmat >, false >
{
public:
    SOFA_CLASS(LUSolver,SOFA_TEMPLATE2(EigenSparseSolver,SOFA_TEMPLATE(Eigen::SparseLU,AssembledSystem::cmat),false));
};


}
}
}



#endif

