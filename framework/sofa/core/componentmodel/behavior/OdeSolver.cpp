#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/tree/MechanicalAction.h>
#include <sofa/simulation/tree/MechanicalVPrintAction.h>


#include <stdlib.h>
#include <math.h>

using namespace sofa::simulation::tree;

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

OdeSolver::OdeSolver()
    : mat(NULL), result(0)
{}

OdeSolver::~OdeSolver()
{}

// void OdeSolver::setGroup(objectmodel::BaseContext* grp)
// {
//     group = grp;
// }



using namespace simulation::tree;

OdeSolver::VectorIndexAlloc::VectorIndexAlloc()
    : maxIndex(VecId::V_FIRST_DYNAMIC_INDEX-1)
{}

unsigned int OdeSolver::VectorIndexAlloc::alloc()
{
    int v;
    if (vfree.empty())
        v = ++maxIndex;
    else
    {
        v = *vfree.begin();
        vfree.erase(vfree.begin());
    }
    vused.insert(v);
    return v;
}








bool OdeSolver::VectorIndexAlloc::free(unsigned int v)
{
    if (v < VecId::V_FIRST_DYNAMIC_INDEX)
        return false;
    // @TODO: Check for errors
    vused.erase(v);
    vfree.insert(v);
    return true;
}

//                 OdeSolver::Context(GNode* node)
//                 : node(node), result(0)
//                 {}

/// Wait for the completion of previous operations and return the result of the last v_dot call
double OdeSolver::finish()
{
    return result;
}

OdeSolver::VecId OdeSolver::v_alloc(VecId::Type t)
{
    VecId v(t, vectors[t].alloc());
    MechanicalVAllocAction(v).execute( getContext() );
    return v;
}

void OdeSolver::v_free(VecId v)
{
    if (vectors[v.type].free(v.index))
        MechanicalVFreeAction(v).execute( getContext() );
}

void OdeSolver::v_clear(VecId v) ///< v=0
{
    MechanicalVOpAction(v).execute( getContext() );
}

void OdeSolver::v_eq(VecId v, VecId a) ///< v=a
{
    MechanicalVOpAction(v,a).execute( getContext() );
}

void OdeSolver::v_peq(VecId v, VecId a, double f) ///< v+=f*a
{
    MechanicalVOpAction(v,v,a,f).execute( getContext() );
}
void OdeSolver::v_teq(VecId v, double f) ///< v*=f
{
    MechanicalVOpAction(v,VecId::null(),v,f).execute( getContext() );
}
void OdeSolver::v_dot(VecId a, VecId b) ///< a dot b ( get result using finish )
{
    result = 0;
    MechanicalVDotAction(a,b,&result).execute( getContext() );
}

void OdeSolver::propagateDx(VecId dx)
{
    MechanicalPropagateDxAction(dx).execute( getContext() );
}

void OdeSolver::projectResponse(VecId dx)
{
    MechanicalApplyConstraintsAction(dx).execute( getContext() );
}

void OdeSolver::addMdx(VecId res, VecId dx)
{
    MechanicalAddMDxAction(res,dx).execute( getContext() );
}

void OdeSolver::integrateVelocity(VecId res, VecId x, VecId v, double dt)
{
    MechanicalVOpAction(res,x,v,dt).execute( getContext() );
}

void OdeSolver::accFromF(VecId a, VecId f)
{
    MechanicalAccFromFAction(a,f).execute( getContext() );
}

void OdeSolver::propagatePositionAndVelocity(double t, VecId x, VecId v)
{
    MechanicalPropagatePositionAndVelocityAction(t,x,v).execute( getContext() );
}

void OdeSolver::computeForce(VecId result)
{
    MechanicalResetForceAction(result).execute( getContext() );
    finish();
    MechanicalComputeForceAction(result).execute( getContext() );
}

void OdeSolver::computeDf(VecId df)
{
    MechanicalResetForceAction(df).execute( getContext() );
    finish();
    MechanicalComputeDfAction(df).execute( getContext() );
}

void OdeSolver::computeAcc(double t, VecId a, VecId x, VecId v)
{
    MultiVector f(this, VecId::force());
    propagatePositionAndVelocity(t, x, v);
    computeForce(f);
    if( this->f_printLog.getValue()==true )
    {
        cerr<<"OdeSolver::computeAcc, f = "<<f<<endl;
    }
    accFromF(a, f);
    projectResponse(a);
}

void OdeSolver::print( VecId v, std::ostream& out )
{
    MechanicalVPrintAction(v,out).execute( getContext() );
}

void OdeSolver::printWithElapsedTime( VecId v,  unsigned time )
{
    MechanicalVPrintWithElapsedTimeAction(v,time).execute( getContext() );
}

//                 double OdeSolver::getTime() const
// {
//     return this->getTime();
// }

// Matrix Computing in ForceField

void OdeSolver::getMatrixDimension(unsigned int * const nbRow, unsigned int * const nbCol)
{
    MechanicalGetMatrixDimensionAction(nbRow, nbCol).execute( getContext() );
}


void OdeSolver::computeMatrix(defaulttype::SofaBaseMatrix *mat, double mFact, double bFact, double kFact, unsigned int offset)
{
    if (mat != NULL)
    {
        MechanicalComputeMatrixAction(mat, mFact, bFact, kFact, offset).execute( getContext() );
    }
}


void OdeSolver::computeOpVector(defaulttype::SofaBaseVector *vect, unsigned int offset)
{
    if (vect != NULL)
    {
        MechanicalComputeVectorAction(vect, offset).execute( getContext() );
    }
}


void OdeSolver::matResUpdatePosition(defaulttype::SofaBaseVector *vect, unsigned int offset)
{
    if (vect != NULL)
    {
        MechanicalMatResUpdatePositionAction(vect, offset).execute( getContext() );
    }
}


void OdeSolver::computeCompliance(double dt, double **W, double *dFree, int &numContact)
{
    MechanicalComputeComplianceAction(dt, W,dFree, numContact).execute(getContext());
}


} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

