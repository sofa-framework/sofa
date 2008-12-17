#ifndef SOFA_COMPONENT_MASTERSOLVER_MASTERCONSTRAINTSOLVER_H
#define SOFA_COMPONENT_MASTERSOLVER_MASTERCONSTRAINTSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/common/MasterSolverImpl.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/component/odesolver/OdeSolverImpl.h>
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
#ifdef DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
        //serr<<"creation of the visitor"<<sendl;
    }

    virtual Result fwdConstraint(simulation::Node* node, core::componentmodel::behavior::BaseConstraint* c)
    {
        //serr<<"fwdConstraint called on "<<c->getName()<<sendl;

        ctime_t t0 = begin(node, c);
        c->getConstraintResolution(_res, _offset);
        end(node, c, t0);
        return RESULT_CONTINUE;
    }

#ifdef DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
private:
    std::vector<core::componentmodel::behavior::ConstraintResolution*>& _res;
    unsigned int _offset;
};

class MechanicalSetConstraint : public simulation::MechanicalVisitor
{
public:
    MechanicalSetConstraint(unsigned int &_contactId)
        :contactId(_contactId)
    {
#ifdef DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual Result fwdConstraint(simulation::Node* node, core::componentmodel::behavior::BaseConstraint* c)
    {
        ctime_t t0 = begin(node, c);
        c->applyConstraint(contactId);
        end(node, c, t0);
        return RESULT_CONTINUE;
    }


    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalSetConstraint"; }

    virtual bool isThreadSafe() const
    {
        return false;
    }
#ifdef DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif

protected:
    unsigned int &contactId;
};

class MechanicalAccumulateConstraint2 : public simulation::MechanicalVisitor
{
public:
    MechanicalAccumulateConstraint2()
    {
#ifdef DUMP_VISITOR_INFO
        setReadWriteVectors();
#endif
    }

    virtual void bwdMechanicalMapping(simulation::Node* node, core::componentmodel::behavior::BaseMechanicalMapping* map)
    {
        ctime_t t0 = begin(node, map);
        map->accumulateConstraint();
        end(node, map, t0);
    }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "MechanicalAccumulateConstraint2"; }

    virtual bool isThreadSafe() const
    {
        return false;
    }
#ifdef DUMP_VISITOR_INFO
    void setReadWriteVectors()
    {
    }
#endif
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

    Data<bool> displayTime;
    Data<double> _tol;
    Data<int> _maxIt;
    Data<bool> doCollisionsFirst;

private:
    void gaussSeidelConstraint(int dim, double* dfree, double** w, double* force, double* d, std::vector<core::componentmodel::behavior::ConstraintResolution*>& res);

    std::vector<core::componentmodel::behavior::BaseConstraintCorrection*> constraintCorrections;

    LPtrFullMatrix<double> _W;
    FullVector<double> _dFree, _force, _d;              // cf. These Duriez
    FullVector<bool> _constraintsType;

    std::vector<core::componentmodel::behavior::ConstraintResolution*> _constraintsResolutions;


};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
