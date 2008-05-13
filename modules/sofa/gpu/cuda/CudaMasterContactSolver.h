#ifndef SOFA_COMPONENT_ODESOLVER_CUDAMASTERCONTACTSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_CUDAMASTERCONTACTSOLVER_H

#include <sofa/simulation/tree/MasterSolverImpl.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
#include <sofa/gpu/cuda/CudaLCP.h>
#include "CudaTypesBase.h"

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::component::linearsolver;
using namespace helper::system::thread;
using namespace sofa::gpu::cuda;


class CudaMechanicalGetConstraintValueVisitor : public simulation::tree::MechanicalVisitor
{
public:
    CudaMechanicalGetConstraintValueVisitor(defaulttype::BaseVector * v): _v(v) {}

    virtual Result fwdConstraint(simulation::Node*,core::componentmodel::behavior::BaseConstraint* c)
    {
        c->getConstraintValue(_v);
        return RESULT_CONTINUE;
    }
private:
    defaulttype::BaseVector * _v;
};

class CudaMechanicalGetContactIDVisitor : public simulation::tree::MechanicalVisitor
{
public:
    CudaMechanicalGetContactIDVisitor(long *id, unsigned int offset = 0)
        : _id(id),_offset(offset) {}

    virtual Result fwdConstraint(simulation::Node*,core::componentmodel::behavior::BaseConstraint* c)
    {
        c->getConstraintId(_id, _offset);
        return RESULT_CONTINUE;
    }

private:
    long *_id;
    unsigned int _offset;
};

template<class real>
class CudaMasterContactSolver : public sofa::simulation::tree::MasterSolverImpl, public virtual sofa::core::objectmodel::BaseObject
{
public:
    typedef real Real;
    Data<bool> initial_guess_d;

    Data < double > tol_d;
    Data<int> maxIt_d;
    Data < double > mu_d;

    Data<int> useGPU_d;

    CudaMasterContactSolver();

    void step (double dt);

    virtual void init();

private:
    std::vector<core::componentmodel::behavior::BaseConstraintCorrection*> constraintCorrections;
    void computeInitialGuess();
    void keepContactForcesValue();

    void build_LCP();

    CudaBaseMatrix<real> _W;
    CudaBaseVector<real> _dFree, _f, _res;

    unsigned int _numConstraints;
    double _mu;
    simulation::tree::GNode *context;

    typedef struct
    {
        Vector3 n;
        Vector3 t;
        Vector3 s;
        Vector3 F;
        long id;

    } contactBuf;

    contactBuf *_PreviousContactList;
    unsigned int _numPreviousContact;
    long *_cont_id_list;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
