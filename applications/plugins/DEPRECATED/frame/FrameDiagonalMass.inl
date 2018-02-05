/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FRAME_FRAMEDIAGONALMASS_INL
#define FRAME_FRAMEDIAGONALMASS_INL

#include <sofa/core/visual/VisualParams.h>
#include "FrameDiagonalMass.h"
#include <sofa/core/behavior/Mass.inl>
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/template.h>
#include <SofaBaseMechanics/AddMToMatrixFunctor.h>
#include <sofa/simulation/Simulation.h>
#include <SofaBaseTopology/TopologyData.inl>

namespace sofa
{

namespace component
{

namespace mass
{

template <class DataTypes, class MassType>
FrameDiagonalMass<DataTypes, MassType>::FrameDiagonalMass()
    : f_mass0 ( initData ( &f_mass0,"f_mass0","vector of lumped blocks of the mass matrix in the rest position." ) )
    , f_mass ( initData ( &f_mass,"f_mass","vector of lumped blocks of the mass matrix." ) )
    , showAxisSize ( initData ( &showAxisSize, 1.0f, "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
    , fileMass( initData(&fileMass,  "fileMass", "File to specify the mass" ) )
    , damping ( initData ( &damping, 0.0f, "damping", "add a force which is \"- damping * speed\"" ) )
{
    this->addAlias(&fileMass,"filename");
}


template <class DataTypes, class MassType>
FrameDiagonalMass<DataTypes, MassType>::~FrameDiagonalMass()
{
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::clear()
{
    MassVector& masses0 = *f_mass0.beginEdit();
    masses0.clear();
    f_mass0.endEdit();
    MassVector& masses = *f_mass.beginEdit();
    masses.clear();
    f_mass.endEdit();
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addMass ( const MassType& m )
{
    MassVector& masses0 = *f_mass0.beginEdit();
    masses0.push_back (m);
    f_mass0.endEdit();
    MassVector& masses = *f_mass.beginEdit();
    masses.push_back (m);
    f_mass.endEdit();
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::resize ( int vsize )
{
    MassVector& masses0 = *f_mass0.beginEdit();
    masses0.resize (vsize);
    f_mass0.endEdit();
    MassVector& masses = *f_mass.beginEdit();
    masses.resize (vsize);
    f_mass.endEdit();
}

// -- Mass interface
template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addMDx ( VecDeriv& res, const VecDeriv& dx, double factor )
{
    const VecCoord& xfrom = *this->mstate->getX();
    const MassVector& vecMass0 = f_mass0.getValue();
    if ( vecMass0.size() != xfrom.size() || frameData->mappingHasChanged) // TODO remove the first condition when mappingHasChanged will be generalized
        updateMass();

    VecDeriv resCpy = res;

    const MassVector& masses = f_mass.getValue();
    if ( factor == 1.0 )
    {
        for ( unsigned int i=0; i<dx.size(); i++ )
        {
            res[i] += dx[i] * masses[i];
            if( this->f_printLog.getValue() )
                serr<<"FrameDiagonalMass<DataTypes, MassType>::addMDx, res = " << res[i] << sendl;
        }
    }
    else
    {
        for ( unsigned int i=0; i<dx.size(); i++ )
        {
            res[i] += ( dx[i]* masses[i] ) * ( Real ) factor; // damping.getValue() * invSqrDT;
            if( this->f_printLog.getValue() )
                serr<<"FrameDiagonalMass<DataTypes, MassType>::addMDx, res = " << res[i] << sendl;
        }
    }
}



template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::accFromF ( VecDeriv& a, const VecDeriv& f )
{

    const MassVector& masses = f_mass.getValue();
    for ( unsigned int i=0; i<f.size(); i++ )
    {
        a[i] = f[i] / masses[i];
    }
}

template <class DataTypes, class MassType>
double FrameDiagonalMass<DataTypes, MassType>::getKineticEnergy ( const VecDeriv& v ) const
{

    const MassVector& masses = f_mass.getValue();
    double e = 0;
    for ( unsigned int i=0; i<masses.size(); i++ )
    {
        e += v[i]*masses[i]*v[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }
    return e/2;
}

template <class DataTypes, class MassType>
double FrameDiagonalMass<DataTypes, MassType>::getPotentialEnergy ( const VecCoord& x ) const
{
    double e = 0;
    const MassVector& masses = f_mass.getValue();
    // gravity
    Vec3 g ( this->getContext()->getGravity() );
    VecIn theGravity;
    theGravity[0]=g[0], theGravity[1]=g[1], theGravity[2]=g[2];
    for ( unsigned int i=0; i<x.size(); i++ )
    {
        VecIn translation;
        translation[0]=(float)x[i].getCenter()[0],  translation[0]=(float)x[1].getCenter()[1], translation[2]=(float)x[i].getCenter()[2];
        const MatInxIn& m = masses[i].inertiaMatrix;
        e -= translation * (m * theGravity);
    }
    return e;
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addMToMatrix ( defaulttype::BaseMatrix * mat, double mFact, unsigned int &offset )
{
    const MassVector& masses = f_mass.getValue();
    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    for ( unsigned int i=0; i<masses.size(); i++ )
        calc ( mat, masses[i], offset + N*i, mFact );
}


template <class DataTypes, class MassType>
double FrameDiagonalMass<DataTypes, MassType>::getElementMass ( unsigned int /*index*/ ) const
{
//  return ( SReal ) ( f_mass.getValue() [index] );
    cerr<<"WARNING : double FrameDiagonalMass<DataTypes, MassType>::getElementMass ( unsigned int index ) const IS NOT IMPLEMENTED" << endl;
    return 0;
}


template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::getElementMass ( unsigned int index, defaulttype::BaseMatrix *m ) const
{
    const unsigned int dimension = defaulttype::DataTypeInfo<Deriv>::size();
    if ( m->rowSize() != dimension || m->colSize() != dimension ) m->resize ( dimension,dimension );

    m->clear();
    const MassVector& masses = f_mass.getValue();
    AddMToMatrixFunctor<Deriv,MassType>() ( m, masses[index], 0, 1 );
}


template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::init()
{
    if (!fileMass.getValue().empty()) load(fileMass.getFullPath().c_str());

    Inherited::init();
}


template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::reinit()
{
    updateMass();
    Inherited::reinit();
}


template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::bwdInit()
{
    // Get the first FramData which is physical
    frameData = NULL;
    vector<FData*> vFData;
    sofa::core::objectmodel::BaseContext* context=  this->getContext();
    context->get<FData>( &vFData, core::objectmodel::BaseContext::SearchDown);
    FData* tmpFData = NULL;
    for ( typename vector<FData *>::iterator it = vFData.begin(); it != vFData.end(); it++)
    {
        tmpFData = (*it);
        if ( tmpFData && tmpFData->isPhysical)
        {
            frameData = tmpFData;
            break;
        }
    }
    if ( ! frameData )
    {
        serr << "Can't find FrameBlendingMapping component." << sendl;
        return;
    }

    updateMass();
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addGravityToV (const core::MechanicalParams* mparams /* PARAMS FIRST */, core::MultiVecDerivId vid)
{
    if ( this->mstate.get(mparams) )
    {
        helper::WriteAccessor< DataVecDeriv > v = *vid[this->mstate.get(mparams)].write();

        // gravity
        Vec3 g ( this->getContext()->getGravity() );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2] );
        Deriv hg = theGravity * ( typename DataTypes::Real ) mparams->dt();

        for ( unsigned int i=0; i<v.size(); i++ )
        {
            v[i] += hg;
        }
    }
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::addForce ( VecDeriv& f, const VecCoord& /*x*/, const VecDeriv& v )
{
    const VecCoord& xfrom = *this->mstate->getX();
    const MassVector& vecMass0 = f_mass0.getValue();
    const MassVector& vecMass = f_mass.getValue();
    if ( vecMass0.size() != xfrom.size() || frameData->mappingHasChanged)
        updateMass();

    rotateMass();

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if ( this->m_separateGravity.getValue() )
        return;

    // gravity
    Vec3 g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2] );

//    // velocity-based stuff
//    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
//    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;
//
//    // project back to local frame
//    vframe = this->getContext()->getPositionInWorld() / vframe;
//    aframe = this->getContext()->getPositionInWorld().backProjectVector ( aframe );

    // add weight and inertia force
    const double& invDt = 1./this->getContext()->getDt();
    for (unsigned int i = 0; i < vecMass.size(); ++i)
    {
        Deriv fDamping = - (vecMass[i] * v[i] * damping.getValue() * invDt);
        f[i] += theGravity*vecMass[i] + fDamping; //  + core::behavior::inertiaForce ( vframe,aframe,masses[i],x[i],v[i] );
    }
    if( this->f_printLog.getValue() )
    {
        serr << "FrameDiagonalMass<DataTypes, MassType>::addForce" << sendl;
        serr << "Masse:" << sendl;
        for(unsigned int i = 0; i < vecMass.size(); ++i)
            serr << i << ": " << vecMass[i].inertiaMatrix << sendl;
        serr << "Force_Masse: " << sendl;
        for(unsigned int i = 0; i < f.size(); ++i)
            serr << i << ": " << f[i] << sendl;
    }
}

template <class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::draw(const core::visual::VisualParams* vparams)
{
    const MassVector& masses = f_mass.getValue();
    if ( !vparams->displayFlags().getShowBehaviorModels() ) return;
    helper::ReadAccessor<VecCoord> x = *this->mstate->getX();
    if ( x.size() != masses.size()) return;
    for ( unsigned int i=0; i<x.size(); i++ )
    {
        vparams->drawTool()->pushMatrix();
        float glTransform[16];
        x[i].writeOpenGlMatrix ( glTransform );
        vparams->drawTool()->multMatrix( glTransform );
        vparams->drawTool()->drawFrame ( Vec3(), Quat(), Vec3 ( 1,1,1 )*showAxisSize.getValue() );
        vparams->drawTool()->popMatrix();
    }
}

template <class DataTypes, class MassType>
class FrameDiagonalMass<DataTypes, MassType>::Loader : public helper::io::MassSpringLoader
{
public:
    FrameDiagonalMass<DataTypes, MassType>* dest;
    Loader ( FrameDiagonalMass<DataTypes, MassType>* dest ) : dest ( dest ) {}
    virtual void addMass ( SReal /*px*/, SReal /*py*/, SReal /*pz*/, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal mass, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/ )
    {
        dest->addMass ( MassType ( ( Real ) mass ) );
    }
};

template <class DataTypes, class MassType>
bool FrameDiagonalMass<DataTypes, MassType>::load ( const char *filename )
{
    clear();
    if ( filename!=NULL && filename[0]!='\0' )
    {
        Loader loader ( this );
        return loader.load ( filename );
    }
    else return false;
}


template<class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::updateMass()
{
    // ReCompute the blocks
    const unsigned int& fromSize = this->mstate->getX()->size();
    this->resize ( fromSize );
    MassVector& mass0 = *f_mass0.beginEdit();
    MassVector& mass  = *f_mass .beginEdit();
    frameData->LumpMassesToFrames(mass0, mass);
    f_mass0.endEdit();
    f_mass .endEdit();
}


template<class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::rotateMass()
{
    const VecCoord& xfrom0 = *this->mstate->getX0();
    const VecCoord& xfrom = *this->mstate->getX();
    const MassVector& vecMass0 = f_mass0.getValue();
    MassVector& vecMass = *f_mass.beginEdit();

    // Rotate the mass
    for ( unsigned int i = 0; i < xfrom.size(); ++i)
    {
        Mat33 relRot;
        computeRelRot( relRot, xfrom[i].getOrientation(), xfrom0[i].getOrientation());
        rotateM( vecMass[i].inertiaMatrix, vecMass0[i].inertiaMatrix, relRot);
        vecMass[i].recalc();
    }
    f_mass.endEdit();
}


template<class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::computeRelRot (Mat33& relRot, const Quat& q, const Quat& q0)
{
    Quat q0_inv;
    q0_inv[0]=-q0[0];
    q0_inv[1]=-q0[1];
    q0_inv[2]=-q0[2];
    q0_inv[3]=q0[3];
    Quat qrel;
    qrel = q * q0_inv;
    QtoR( relRot, qrel);
}


template<class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::rotateM( MatInxIn& M, const MatInxIn& M0, const Mat33& R)
{
    int i,j,k;

    MatInxIn M1;
    M1.fill(0);
    M.fill(0);
    for (i=0; i<3; i++) for (j=0; j<3; j++) for (k=0; k<3; k++)
            {
                M1[i][j]+=R[i][k]*M0[k][j];
                M1[i][j+3]+=R[i][k]*M0[k][j+3];
                M1[i+3][j]+=R[i][k]*M0[k+3][j];
                M1[i+3][j+3]+=R[i][k]*M0[k+3][j+3];
            }
    for (i=0; i<3; i++) for (j=0; j<3; j++) for (k=0; k<3; k++)
            {
                M[i][j]+=M1[i][k]*R[j][k];
                M[i][j+3]+=M1[i][k+3]*R[j][k];
                M[i+3][j]+=M1[i+3][k]*R[j][k];
                M[i+3][j+3]+=M1[i+3][k+3]*R[j][k];
            }
}

template<class DataTypes, class MassType>
void FrameDiagonalMass<DataTypes, MassType>::QtoR( Mat33& M, const sofa::helper::Quater<Real>& q)
{
// q to M
    Real xs = q[0]*(Real)2., ys = q[1]*(Real)2., zs = q[2]*(Real)2.;
    Real wx = q[3]*xs, wy = q[3]*ys, wz = q[3]*zs;
    Real xx = q[0]*xs, xy = q[0]*ys, xz = q[0]*zs;
    Real yy = q[1]*ys, yz = q[1]*zs, zz = q[2]*zs;
    M[0][0] = (Real)1.0 - (yy + zz);
    M[0][1]= xy - wz;
    M[0][2] = xz + wy;
    M[1][0] = xy + wz;
    M[1][1] = (Real)1.0 - (xx + zz);
    M[1][2] = yz - wx;
    M[2][0] = xz - wy;
    M[2][1] = yz + wx;
    M[2][2] = (Real)1.0 - (xx + yy);
}


} // namespace mass

} // namespace component

} // namespace sofa

#endif
