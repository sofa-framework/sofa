/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FRAME_FRAMEVOLUMEPRESERVATIONFORCEFIELD_INL
#define FRAME_FRAMEVOLUMEPRESERVATIONFORCEFIELD_INL


#include <sofa/core/behavior/ForceField.inl>
#include "FrameVolumePreservationForceField.h"
#include "DeformationGradientTypes.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/Mapping.inl>
/*
*/
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace forcefield
{


using namespace helper;


template <class DataTypes>
FrameVolumePreservationForceField<DataTypes>::FrameVolumePreservationForceField(core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit1(mm)
{
}

template <class DataTypes>
FrameVolumePreservationForceField<DataTypes>::~FrameVolumePreservationForceField()
{}


template <class DataTypes>
void FrameVolumePreservationForceField<DataTypes>::init()
{
    Inherit1::init();
    core::objectmodel::BaseContext* context = this->getContext();
    sampleData = context->get<SampleData>();
    if( sampleData==NULL )
    {
        cerr<<"FrameVolumePreservationForceField<DataTypes>::init(), sampledata not found"<< endl;
    }

    material = context->get<Material>();
    if( material==NULL )
    {
        cerr<<"FrameVolumePreservationForceField<DataTypes>::init(), material not found"<< endl;
    }

    ReadAccessor<Data<VecCoord> > out (*this->getMState()->read(core::ConstVecCoordId::restPosition()));
    this->bulkModulus.resize(out.size());

    for(unsigned int i=0; i<out.size(); i++) // treat each sample
    {
        if(material) this->bulkModulus[i]=(Real)material->getBulkModulus(i);
        else this->bulkModulus[i]=(Real)1000;
        if( this->f_printLog.getValue() )
            std::cout<<"FrameVolumePreservationForceField<DataTypes>::bulkModulus["<<i<<"]"<<bulkModulus[i]<<std::endl;
    }



    this->volume.resize(out.size());
    for(unsigned int i=0; i<out.size(); i++) // treat each sample
    {
        typename DataTypes::SpatialCoord point;
        DataTypes::get(point[0],point[1],point[2], out[i]) ;
        if(material)
        {
            vector<Real> moments;
            material->computeVolumeIntegrationFactors(i,point,0,moments);
            this->volume[i]=moments[0];
        }
        else this->volume[i]=1; // default volume when there is no material
        if( this->f_printLog.getValue() )
            std::cout<<"FrameVolumePreservationForceField<DataTypes>::volume["<<i<<"]"<<volume[i]<<std::endl;
    }
}

#define MIN_DETERMINANT  1.0e-2

template <class DataTypes>
void FrameVolumePreservationForceField<DataTypes>::addForce(DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& /*_v */, const core::MechanicalParams* /*mparams*/)
{
    ReadAccessor<DataVecCoord> x(_x);
    WriteAccessor<DataVecDeriv> f(_f);

    ddet.resize(x.size());
    det.resize(x.size());
    for(unsigned int i=0; i<x.size(); i++)
    {
        det[i]=determinant(x[i].getMaterialFrame());
        if(det[i]>MIN_DETERMINANT || det[i]<-MIN_DETERMINANT)
        {
            invertMatrix(ddet[i],x[i].getMaterialFrame());
            ddet[i].transpose();
            // energy W=vol.k[det-1]^2/2
            ddet[i]*=det[i];
            f[i].getMaterialFrame()-=ddet[i]*volume[i]*bulkModulus[i]*(det[i]-(Real)1.); // force f=-vol.k[det-1]ddet
            // energy W=vol.k[log|det|^2]/2
//						f[i].getMaterialFrame()-=ddet[i]*volume[i]*bulkModulus[i]*log(fabs(det[i])); // force f=-vol.k.log|det|.ddet
        }
        else ddet[i].fill(0);

        if( this->f_printLog.getValue() )
            if(i==100)
            {
                sout<<"FrameVolumePreservationForceField<DataTypes>::addForce, F = " << x[i].getMaterialFrame() << sendl;
                sout<<"FrameVolumePreservationForceField<DataTypes>::addForce, det = " << det[i] << sendl;
                sout<<"FrameVolumePreservationForceField<DataTypes>::addForce, ddet = " << ddet[i] << sendl;
                sout<<"FrameVolumePreservationForceField<DataTypes>::addForce, stress deformation gradient form = " << f[i] << sendl;
            }
    }
}

template <class DataTypes>
void FrameVolumePreservationForceField<DataTypes>::addDForce(DataVecDeriv& _df , const DataVecDeriv&  _dx , const core::MechanicalParams* mparams)
{
    ReadAccessor<DataVecCoord> x (*this->getMState()->read(core::ConstVecCoordId::position()));
    ReadAccessor<DataVecDeriv> dx(_dx);
    WriteAccessor<DataVecDeriv> df(_df);

    Real kFactor = (Real)mparams->kFactor();

    Real det2; Frame ddet2,x2;
    for(unsigned int i=0; i<dx.size(); i++)
    {
        x2=x[i].getMaterialFrame()+dx[i].getMaterialFrame();
        det2=determinant(x2);
        if(det2>MIN_DETERMINANT || det2<-MIN_DETERMINANT)
        {
            invertMatrix(ddet2,x2);
            ddet2.transpose();
            ddet2*=det2;
            df[i].getMaterialFrame()+=(ddet[i]*(det[i]-(Real)1.)-ddet2*(det2-(Real)1.))*kFactor*volume[i]*bulkModulus[i];  // force f=-vol.k[det-1]ddet
//					df[i].getMaterialFrame()+=(ddet[i]*log(fabs(det[i]))-ddet2*log(fabs(det2)))*kFactor*volume[i]*bulkModulus[i];   // force f=-vol.k.log|det|.ddet
        }
        else ddet2.fill(0);

        if( this->f_printLog.getValue() )
            if(i==100)
            {
                sout<<"FrameVolumePreservationForceField<DataTypes>::addDForce, dF = " << dx[i].getMaterialFrame() << sendl;
                sout<<"FrameVolumePreservationForceField<DataTypes>::addDForce, det = " << det[i] << sendl;
                sout<<"FrameVolumePreservationForceField<DataTypes>::addDForce, ddet = " << ddet[i] << sendl;
                sout<<"FrameVolumePreservationForceField<DataTypes>::addDForce, stress deformation gradient change after accumulating "<< kFactor<<"* df = " << df[i] << sendl;
            }
    }


    /*
    // derivative of the force based on the energy vol.k[det-1]^2/2 :  df = vol.k.[ (1-1/det)ddet.dF^T.ddet  - (2-1/det)ddet.Trace(dF^T.ddet) ]
    Real trace,invdet;
    Frame M;
    for(unsigned int i=0; i<dx.size(); i++)
    {
    	if(det[i]>MIN_DETERMINANT || det[i]<-MIN_DETERMINANT)	{
    	invdet=(Real)1./det[i];
    	trace=0;
    	for(unsigned int k=0; k<material_dimensions; k++) for(unsigned int l=0; l<material_dimensions; l++) trace+=dx[i].getMaterialFrame()[k][l]*ddet[i][k][l];
    	trace*=((Real)2.-invdet);
    	M=ddet[i]*dx[i].getMaterialFrame().multTranspose(ddet[i])*((Real)1.-invdet);
    	M-=ddet[i]*trace;
    	df[i].getMaterialFrame()+=volume[i]*bulkModulus[i]*kFactor*M;
    	if(i==100)
    	{
            sout<<"trace = " << trace << sendl;
            sout<<"M = " << volume[i]*bulkModulus[i]*kFactor*M << sendl;
    	}
    	}
    	if( this->f_printLog.getValue() )
    	if(i==100)
    	{
            sout<<"FrameVolumePreservationForceField<DataTypes>::addDForce, dF = " << dx[i].getMaterialFrame() << sendl;
            sout<<"FrameVolumePreservationForceField<DataTypes>::addDForce, det = " << det[i] << sendl;
            sout<<"FrameVolumePreservationForceField<DataTypes>::addDForce, ddet = " << ddet[i] << sendl;
            sout<<"FrameVolumePreservationForceField<DataTypes>::addDForce, stress deformation gradient change after accumulating "<< kFactor<<"* df = " << df[i] << sendl;
        }
    }
    */
}
}
}
} // namespace sofa

#endif
