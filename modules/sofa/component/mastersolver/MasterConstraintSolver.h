#ifndef SOFA_COMPONENT_MASTERSOLVER_MASTERCONSTRAINTSOLVER_H
#define SOFA_COMPONENT_MASTERSOLVER_MASTERCONSTRAINTSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/tree/MasterSolverImpl.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>
#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/tree/OdeSolverImpl.h>
#include <sofa/component/linearsolver/FullMatrix.h>

namespace sofa
{

namespace component
{

namespace mastersolver
{

using namespace sofa::defaulttype;
using namespace sofa::component::linearsolver;
using namespace helper::system::thread;

class MechanicalGetConstraintTypeVisitor : public simulation::tree::MechanicalVisitor
{
public:
    MechanicalGetConstraintTypeVisitor(bool *type, unsigned int offset = 0)
        : _type(type),_offset(offset)
    {
    }

    virtual Result fwdConstraint(simulation::Node* /*node*/, core::componentmodel::behavior::BaseConstraint* c)
    {
        c->getConstraintType(_type, _offset);
        return RESULT_CONTINUE;
    }

private:
    bool *_type;
    unsigned int _offset;
};

class MasterConstraintSolver : public sofa::simulation::tree::MasterSolverImpl//, public sofa::simulation::tree::OdeSolverImpl
{
public:

    MasterConstraintSolver();
    ~MasterConstraintSolver();
    // virtual const char* getTypeName() const { return "MasterSolver"; }

    void step (double dt);

    //virtual void propagatePositionAndVelocity(double t, VecId x, VecId v);

    virtual void init();

private:
    void gaussSeidelConstraint(int, double *, double **, double *, bool *);

    std::vector<core::componentmodel::behavior::BaseConstraintCorrection*> constraintCorrections;

    LPtrFullMatrix<double> _W;
    FullVector<double> _dFree, _result;
    FullVector<bool> _constraintsType;

    Data<double> _tol, _mu;
    Data<int> _maxIt;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
