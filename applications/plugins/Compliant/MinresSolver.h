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

/** Variant of ComplianceSolver where a minres iterative solver is used in place of the direct solver.
*/
class SOFA_Compliant_API MinresSolver  : public ComplianceSolver
{
public:
    SOFA_CLASS(MinresSolver, ComplianceSolver );

    virtual void solveEquation();

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

//    // useful one-liners
//    const mat& M() const;
//    mat& M();

//    const mat& J() const;
//    mat& J();

//    const mat& C() const;
//    mat& C();

    const mat& projMatrix() const;
//    mat& projMatrix();

//    // same
//    vec& f();
//    const vec& f() const;

//    const vec& phi() const;
//    vec& phi();


    // solver type
    typedef ::minres<SReal> minres;

    // requires visitor.fetch() == true
    vec solve_schur(minres::params& ) ;
    vec solve_kkt(minres::params& ) ;

    // internal types
    struct schur;
    struct kkt;
};





}
}
}

#endif // SOFA_COMPONENT_ODESOLVER_ComplianceSolver_H
