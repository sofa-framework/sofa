/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
//#include <sofa/simulation/tree/MechanicalVisitor.h>
//#include <sofa/simulation/tree/MechanicalVPrintVisitor.h>


#include <stdlib.h>
#include <math.h>

//using namespace sofa::simulation::tree;

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

OdeSolver::OdeSolver()
//: /*mat(NULL),*/ result(0)
{}

OdeSolver::~OdeSolver()
{}

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

#if 0

using namespace simulation::tree;

double OdeSolver::finish()
{
    return result;
}

OdeSolver::VecId OdeSolver::v_alloc(VecId::Type t)
{
    VecId v(t, vectors[t].alloc());
    MechanicalVAllocVisitor(v).execute( getContext() );
    return v;
}

void OdeSolver::v_free(VecId v)
{
    if (vectors[v.type].free(v.index))
        MechanicalVFreeVisitor(v).execute( getContext() );
}

void OdeSolver::v_clear(VecId v) ///< v=0
{
    MechanicalVOpVisitor(v).execute( getContext() );
}

void OdeSolver::v_eq(VecId v, VecId a) ///< v=a
{
    MechanicalVOpVisitor(v,a).execute( getContext() );
}

void OdeSolver::v_peq(VecId v, VecId a, double f) ///< v+=f*a
{
    MechanicalVOpVisitor(v,v,a,f).execute( getContext() );
}
void OdeSolver::v_teq(VecId v, double f) ///< v*=f
{
    MechanicalVOpVisitor(v,VecId::null(),v,f).execute( getContext() );
}
void OdeSolver::v_dot(VecId a, VecId b) ///< a dot b ( get result using finish )
{
    result = 0;
    MechanicalVDotVisitor(a,b,&result).execute( getContext() );
}

void OdeSolver::propagateDx(VecId dx)
{
    MechanicalPropagateDxVisitor(dx).execute( getContext() );
}

void OdeSolver::projectResponse(VecId dx, double **W)
{
    MechanicalApplyConstraintsVisitor(dx, W).execute( getContext() );
}

void OdeSolver::addMdx(VecId res, VecId dx)
{
    MechanicalAddMDxVisitor(res,dx).execute( getContext() );
}

void OdeSolver::integrateVelocity(VecId res, VecId x, VecId v, double dt)
{
    MechanicalVOpVisitor(res,x,v,dt).execute( getContext() );
}

void OdeSolver::accFromF(VecId a, VecId f)
{
    MechanicalAccFromFVisitor(a,f).execute( getContext() );
}

void OdeSolver::propagatePositionAndVelocity(double t, VecId x, VecId v)
{
    MechanicalPropagatePositionAndVelocityVisitor(t,x,v).execute( getContext() );
}

void OdeSolver::computeForce(VecId result)
{
    MechanicalResetForceVisitor(result).execute( getContext() );
    finish();
    MechanicalComputeForceVisitor(result).execute( getContext() );
}

void OdeSolver::computeDf(VecId df)
{
    MechanicalResetForceVisitor(df).execute( getContext() );
    finish();
    MechanicalComputeDfVisitor(df).execute( getContext() );
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

void OdeSolver::computeContactForce(VecId result)
{
    MechanicalResetForceVisitor(result).execute( getContext() );
    finish();
    MechanicalComputeContactForceVisitor(result).execute( getContext() );
}

void OdeSolver::computeContactDf(VecId df)
{
    MechanicalResetForceVisitor(df).execute( getContext() );
    finish();
    //MechanicalComputeDfVisitor(df).execute( getContext() );
}

void OdeSolver::computeContactAcc(double t, VecId a, VecId x, VecId v)
{
    MultiVector f(this, VecId::force());
    propagatePositionAndVelocity(t, x, v);
    computeContactForce(f);
    if( this->f_printLog.getValue()==true )
    {
        cerr<<"OdeSolver::computeContactAcc, f = "<<f<<endl;
    }

    accFromF(a, f);
    projectResponse(a);
}

void OdeSolver::print( VecId v, std::ostream& out )
{
    MechanicalVPrintVisitor(v,out).execute( getContext() );
}

void OdeSolver::printWithElapsedTime( VecId v,  unsigned time, std::ostream& out )
{
    const double fact = 1000000.0 / (100*helper::system::thread::CTime::getTicksPerSec());
    MechanicalVPrintWithElapsedTimeVisitor(v,(int)((fact*time+0.5)*0.001), out).execute( getContext() );
}

// BaseMatrix & BaseVector Computations
void OdeSolver::addMBK_ToMatrix(defaulttype::BaseMatrix *A, double mFact, double bFact, double kFact, unsigned int offset)
{
    if (A != NULL)
    {
        MechanicalAddMBK_ToMatrixVisitor(A, mFact, bFact, kFact, offset).execute( getContext() );
    }
}

void OdeSolver::addMBKdx_ToVector(VecId res, VecId dx, double mFact, double bFact, double kFact)
{
    MechanicalAddMBKdx_ToVectorVisitor(res, dx, mFact, bFact, kFact);
}

void OdeSolver::multiVector2BasicVector(VecId src, defaulttype::BaseVector *dest, unsigned int offset)
{
    if (dest != NULL)
    {
        MechanicalMultiVector2BasicVectorVisitor(src, dest, offset);
    }
}

void OdeSolver::getMatrixDimension(unsigned int * const nbRow, unsigned int * const nbCol)
{
    MechanicalGetMatrixDimensionVisitor(nbRow, nbCol).execute( getContext() );
}

/*
void OdeSolver::computeMatrix(defaulttype::SofaBaseMatrix *mat, double mFact, double bFact, double kFact, unsigned int offset)
{
    if (mat != NULL)
    {
        MechanicalComputeMatrixVisitor(mat, mFact, bFact, kFact, offset).execute( getContext() );
    }
}


void OdeSolver::computeOpVector(defaulttype::SofaBaseVector *vect, unsigned int offset)
{
    if (vect != NULL)
    {
        MechanicalComputeVectorVisitor(vect, offset).execute( getContext() );
    }
}


void OdeSolver::matResUpdatePosition(defaulttype::SofaBaseVector *vect, unsigned int offset)
{
    if (vect != NULL)
    {
        MechanicalMatResUpdatePositionVisitor(vect, offset).execute( getContext() );
    }
}
*/

#endif

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

