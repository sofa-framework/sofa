/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef FRAME_GREENLAGRANGEFORCEFIELD_INL
#define FRAME_GREENLAGRANGEFORCEFIELD_INL


#include <sofa/core/behavior/ForceField.inl>
#include "GreenLagrangeForceField.h"
#include "DeformationGradientTypes.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/ForceField.inl>
/*
NB fjourdes: I don t get why this include is required to stop
warning C4661 occurences while compiling GreenLagrangeForceField.cpp
*/
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
GreenLagrangeForceField<DataTypes>::GreenLagrangeForceField(core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit1(mm)
{}

template <class DataTypes>
GreenLagrangeForceField<DataTypes>::~GreenLagrangeForceField()
{}


template <class DataTypes>
void GreenLagrangeForceField<DataTypes>::init()
{
    Inherit1::init();
    core::objectmodel::BaseContext* context = this->getContext();
    sampleData = context->get<SampleData>();
    if( sampleData==NULL )
    {
        cerr<<"GreenLagrangeForceField<DataTypes>::init(), sampledata not found"<< endl;
    }



    material = context->get<Material>();
    if( material==NULL )
    {
        cerr<<"GreenLagrangeForceField<DataTypes>::init(), material not found"<< endl;
    }


    ReadAccessor<Data<VecCoord> > out (*this->getMState()->read(core::ConstVecCoordId::restPosition()));

    this->integFactors.resize(out.size());


    for(unsigned int i=0; i<out.size(); i++) // treat each sample
    {
        typename DataTypes::SpatialCoord point;
        DataTypes::get(point[0],point[1],point[2], out[i]) ;
        if(material)
        {
            vector<Real> moments;
            material->computeVolumeIntegrationFactors(i,point,StrainType::strainenergy_order,moments);  // lumpMoments
            for(unsigned int j=0; j<moments.size() && j<this->integFactors[i].size() ; j++)
                this->integFactors[i][j]=moments[j];
        }
        else 	this->integFactors[i][0]=1; // default value when there is no material

        if( this->f_printLog.getValue() )
            std::cout<<"GreenLagrangeForceField<DataTypes>::IntegFactor["<<i<<"](coord "<<point<<")="<<integFactors[i]<<std::endl;
    }
}


template <class DataTypes>
void GreenLagrangeForceField<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v)
{
    ReadAccessor<DataVecCoord> x(_x);
    ReadAccessor<DataVecDeriv> v(_v);
    WriteAccessor<DataVecDeriv> f(_f);
    ReadAccessor<Data<VecMaterialCoord> > out (sampleData->f_materialPoints);

    updateFF( _x.getValue().size());

    // compute strains and strain rates
    for(unsigned int i=0; i<x.size(); i++)
    {
        StrainType::apply(x[i], strain[i]/*,&rotation[i]*/);
        StrainType::mult(v[i], x[i], strainRate[i]/*,&rotation[i]*/);
        if( this->f_printLog.getValue() )
        {
            cerr<<"GreenLagrangeForceField<DataTypes>::addForce, deformation gradient = " << x[i] << endl;
            cerr<<"GreenLagrangeForceField<DataTypes>::addForce, strain = " << strain[i] << endl;
        }
    }
    material->computeStress( stress, &stressStrainMatrices, strain, strainRate, out.ref() );

    // integrate and compute force
    for(unsigned int i=0; i<x.size(); i++)
    {
        StrainType::addMultTranspose(f[i], x[i], stress[i], this->integFactors[i]/*, &rotation[i]*/);
        if( this->f_printLog.getValue() )
        {
            cerr<<"GreenLagrangeForceField<DataTypes>::addForce, stress = " << stress[i] << endl;
            cerr<<"GreenLagrangeForceField<DataTypes>::addForce, stress deformation gradient form= " << f[i] << endl;
        }
    }
}

template <class DataTypes>
void GreenLagrangeForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& _df , const DataVecDeriv&  _dx)
{
    ReadAccessor<DataVecCoord> x (*this->getMState()->read(core::ConstVecCoordId::position()));
    ReadAccessor<DataVecDeriv> dx(_dx);
    WriteAccessor<DataVecDeriv> df(_df);
    strainChange.resize(dx.size());
    stressChange.resize(dx.size());
    ReadAccessor<Data<VecMaterialCoord> > out (sampleData->f_materialPoints);

    // compute strains changes
    for(unsigned int i=0; i<dx.size(); i++)
    {
        StrainType::mult(dx[i], x[i], strainRate[i]/*,&rotation[i]*/);
        if( this->f_printLog.getValue() )
        {
            cerr<<"GreenLagrangeForceField<DataTypes>::addDForce, deformation gradient change = " << dx[i] << endl;
            cerr<<"GreenLagrangeForceField<DataTypes>::addDForce, strain change = " << strainRate[i] << endl;
            cerr<<"GreenLagrangeForceField<DataTypes>::addDForce, stress deformation gradient change before accumulating = " << df[i] << endl;
        }
    }

    // compute stress changes
    material->computeStressChange( stressChange, strainRate, out.ref() );

    // apply factor
    Real kFactor = (Real)mparams->kFactor();
    for(unsigned int i=0; i<dx.size(); i++)
    {
        if( this->f_printLog.getValue() )
        {
            cerr<<"GreenLagrangeForceField<DataTypes>::addDForce, stress change = " << stressChange[i] << endl;
        }
        StrainType::mult(stressChange[i], kFactor);
    }

    // integrate and compute force
    for(unsigned int i=0; i<dx.size(); i++)
    {
        StrainType::addMultTranspose(df[i], x[i], stressChange[i], this->integFactors[i]/*, &rotation[i]*/);
        if( this->f_printLog.getValue() )
        {
            cerr<<"GreenLagrangeForceField<DataTypes>::addDForce, stress deformation gradient change after accumulating "<< kFactor<<"* df = " << df[i] << endl;
        }
    }
}



template <class DataTypes>
void GreenLagrangeForceField<DataTypes>::updateFF (const unsigned int& size)
{
    ReadAccessor<Data<VecCoord> > out (*this->getMState()->read(core::ConstVecCoordId::restPosition()));

    if (stressStrainMatrices.size() == size && this->integFactors.size() == out.size() && !sampleData->mappingHasChanged) return;

    serr << "recompute stiffness matrix" << sendl;

    stressStrainMatrices.resize(size);
    //rotation.resize(size);
    strain.resize(size);
    strainRate.resize(size);
    stress.resize(size);

    this->integFactors.resize(out.size());


    for(unsigned int i=0; i<out.size(); i++) // treat each sample
    {
        typename DataTypes::SpatialCoord point;
        DataTypes::get(point[0],point[1],point[2], out[i]) ;
        if(material)
        {
            vector<Real> moments;
            material->computeVolumeIntegrationFactors(i,point,StrainType::strainenergy_order,moments);  // lumpMoments
            for(unsigned int j=0; j<moments.size() && j<this->integFactors[i].size() ; j++)
                this->integFactors[i][j]=moments[j];
        }
        else
            this->integFactors[i][0]=1; // default value when there is no material

        if( this->f_printLog.getValue() )
            std::cout<<"GreenLagrangeForceField<DataTypes>::IntegFactor["<<i<<"](coord "<<point<<")="<<integFactors[i]<<std::endl;
    }
}

}
}
} // namespace sofa

#endif
