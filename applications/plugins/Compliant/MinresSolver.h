#ifndef SOFA_COMPONENT_ODESOLVER_MinresSolver_H
#define SOFA_COMPONENT_ODESOLVER_MinresSolver_H

#include "ComplianceSolver.h"

#include "utils/minres.h"

namespace sofa
{
using helper::vector;

namespace component
{

namespace odesolver
{

/** Variant of ComplianceSolver where the equation system is solved using MinRes algorithm.

\sa \ref sofa::component::odesolver::ComplianceSolver



*/
class SOFA_Compliant_API MinresSolver  : public ComplianceSolver
{
public:
    SOFA_CLASS(MinresSolver, ComplianceSolver );

    virtual void bwdInit();

    virtual void solve(const core::ExecParams* params, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);

    MinresSolver();

protected:

    typedef SMatrix mat;
    typedef VectorSofa::VectorEigen vec;
    typedef SReal real;

    Data<bool> use_kkt;
    Data<unsigned int> max_iterations;
    Data<unsigned int> iterations_performed;
    Data<real> precision;
    Data<bool> use_warm;

    mutable vec last;

    // fills a solution vector @x with @last solution based on
    // @use_warm value
    void warm(vec& x) const;

    // sends visitor to fetch all data needed. returns true iff data
    // actually fetched (i.e. there is something to solve)
    bool fetch_data(core::MechanicalParams*);

    // useful one-liners
    const mat& M() const;
    mat& M();

    const mat& J() const;
    mat& J();

    const mat& C() const;
    mat& C();

    const mat& P() const;
    mat& P();

    // same
    vec& f();
    const vec& f() const;

    const vec& phi() const;
    vec& phi();

    // friendly api for assembly visitor
    struct visitor
    {
        MatrixAssemblyVisitor assembly;
        MinresSolver* solver;

        visitor(core::MechanicalParams* cparams, MinresSolver* );

        // sends assembly visitor, returns true when solving needs to be
        // done
        bool fetch();

        // dispatches f vector among dofs
        void distribute();

    };

    // convenience constructor
    visitor make_visitor(core::MechanicalParams* );

    // solver type
    typedef ::minres<SReal> minres;

    // requires visitor.fetch() == true
    vec solve_schur(real dt, minres::params& );
    vec solve_kkt(real dt, minres::params& );

    // internal types
    struct schur;
    struct kkt;
};





}
}
}

#endif // SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
