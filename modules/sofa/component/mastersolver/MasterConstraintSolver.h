#ifndef SOFA_COMPONENT_MASTERSOLVER_MASTERCONSTRAINTSOLVER_H
#define SOFA_COMPONENT_MASTERSOLVER_MASTERCONSTRAINTSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/common/MasterSolverImpl.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/common/OdeSolverImpl.h>
#include <sofa/component/linearsolver/FullMatrix.h>

#include <vector>

namespace sofa
{

namespace component
{

namespace mastersolver
{

using namespace sofa::defaulttype;
using namespace sofa::component::linearsolver;
using namespace helper::system::thread;

class MechanicalGetConstraintResolutionVisitor : public simulation::MechanicalVisitor
{
public:
    MechanicalGetConstraintResolutionVisitor(std::vector<core::componentmodel::behavior::ConstraintResolution*>& res, unsigned int offset = 0)
        : _res(res),_offset(offset)
    {
    }

    virtual Result fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
    {
        c->getConstraintResolution(_res, _offset);
        return RESULT_CONTINUE;
    }

private:
    std::vector<core::componentmodel::behavior::ConstraintResolution*>& _res;
    unsigned int _offset;
};

class MasterConstraintSolver : public sofa::simulation::MasterSolverImpl//, public sofa::simulation::tree::OdeSolverImpl
{
public:

    MasterConstraintSolver();
    virtual ~MasterConstraintSolver();
    // virtual const char* getTypeName() const { return "MasterSolver"; }

    void step(double dt);

    //virtual void propagatePositionAndVelocity(double t, VecId x, VecId v);

    virtual void init();

private:
    void gaussSeidelConstraint(int dim, double* dfree, double** w, double* force, double* d, std::vector<core::componentmodel::behavior::ConstraintResolution*>& res);

    std::vector<core::componentmodel::behavior::BaseConstraintCorrection*> constraintCorrections;

    LPtrFullMatrix<double> _W;
    FullVector<double> _dFree, _force, _d;		// cf. These Duriez
    FullVector<bool> _constraintsType;

    std::vector<core::componentmodel::behavior::ConstraintResolution*> _constraintsResolutions;

    Data<double> _tol, _mu;
    Data<int> _maxIt;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
