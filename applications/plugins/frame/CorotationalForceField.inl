#include "CorotationalForceField.h"
#include "DeformationGradientTypes.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/ForceField.inl>
/*
NB fjourdes: I don t get why this include is required to stop
warning C4661 occurences while compiling CorotationalForceField.cpp
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
CorotationalForceField<DataTypes>::CorotationalForceField(core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit1(mm)
{}

template <class DataTypes>
CorotationalForceField<DataTypes>::~CorotationalForceField()
{}


template <class DataTypes>
void CorotationalForceField<DataTypes>::init()
{
    Inherit1::init();
    core::objectmodel::BaseContext* context = this->getContext();
    sampleData = context->get<SampleData>();
    if( sampleData==NULL )
    {
        cerr<<"CorotationalForceField<DataTypes>::init(), material not found"<< endl;
    }



    material = context->get<Material>();
    if( material==NULL )
    {
        cerr<<"CorotationalForceField<DataTypes>::init(), material not found"<< endl;
    }


    ReadAccessor<Data<VecCoord> > out (*this->getMState()->read(core::ConstVecCoordId::restPosition()));

    this->sampleInteg.resize(out.size());


    for(unsigned int i=0; i<out.size(); i++) // treat each sample
    {
        typename DataTypes::SpatialCoord point;
        DataTypes::get(point[0],point[1],point[2], out[i]) ;
        if(material)
        {
            vector<Real> moments;
            material->computeVolumeIntegrationFactors(i,point,StrainType::strain_order,moments);  // lumpMoments
            for(unsigned int j=0; j<moments.size() && j<this->sampleInteg[i].size() ; j++)
                this->sampleInteg[i][j]=moments[j];
        }
        else this->sampleInteg[i][0]=1; // default value for the volume when model vertices are used as gauss points
    }

    //for(unsigned int i=0;i<out.size();i++) std::cout<<"IntegVector["<<i<<"]="<<sampleInteg[i]<<std::endl;

}


template <class DataTypes>
void CorotationalForceField<DataTypes>::addForce(DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v , const core::MechanicalParams* /*mparams*/)
{
    ReadAccessor<DataVecCoord> x(_x);
    ReadAccessor<DataVecDeriv> v(_v);
    WriteAccessor<DataVecDeriv> f(_f);
    ReadAccessor<Data<VecMaterialCoord> > out (sampleData->f_materialPoints);
    stressStrainMatrices.resize(x.size());
    rotation.resize(x.size());
    strain.resize(x.size());
    strainRate.resize(x.size());
    stress.resize(x.size());

    // compute strains and strain rates
    for(unsigned i=0; i<x.size(); i++)
    {
        StrainType::apply(x[i], strain[i],&rotation[i]);
        StrainType::mult(v[i], strainRate[i],&rotation[i]);
        if( this->f_printLog.getValue() )
        {
            cerr<<"CorotationalForceField<DataTypes>::addForce, deformation gradient = " << x[i] << endl;
            cerr<<"CorotationalForceField<DataTypes>::addForce, strain = " << strain[i] << endl;
        }
    }
    material->computeStress( stress, &stressStrainMatrices, strain, strainRate, out.ref() );

    // integrate and compute force
    for(unsigned i=0; i<x.size(); i++)
    {
        StrainType::addMultTranspose(f[i], x[i], stress[i], this->sampleInteg[i], &rotation[i]);
        if( this->f_printLog.getValue() )
        {
            cerr<<"CorotationalForceField<DataTypes>::addForce, stress = " << stress[i] << endl;
            cerr<<"CorotationalForceField<DataTypes>::addForce, stress deformation gradient form= " << f[i] << endl;
        }
    }
}

template <class DataTypes>
void CorotationalForceField<DataTypes>::addDForce(DataVecDeriv& _df , const DataVecDeriv&  _dx , const core::MechanicalParams* mparams)
{
    ReadAccessor<DataVecCoord> dx(_dx);
    WriteAccessor<DataVecDeriv> df(_df);
    strainChange.resize(dx.size());
    stressChange.resize(dx.size());
    ReadAccessor<Data<VecMaterialCoord> > out (sampleData->f_materialPoints);

    // compute strains changes
    for(unsigned i=0; i<dx.size(); i++)
    {
        StrainType::mult(dx[i], strainRate[i]);
        if( this->f_printLog.getValue() )
        {
            cerr<<"CorotationalForceField<DataTypes>::addDForce, deformation gradient change = " << dx[i] << endl;
            cerr<<"CorotationalForceField<DataTypes>::addDForce, strain change = " << strainRate[i] << endl;
            cerr<<"CorotationalForceField<DataTypes>::addDForce, stress deformation gradient change before accumulating = " << df[i] << endl;
        }
    }

    // Todo: apply stiffness matrix and integration factors, compute frame

    material->computeStressChange( stressChange, strainRate, out.ref() );

    // apply factor
    Real kFactor = (Real)mparams->kFactor();
    for(unsigned i=0; i<dx.size(); i++)
    {
        if( this->f_printLog.getValue() )
        {
            cerr<<"CorotationalForceField<DataTypes>::addDForce, stress change = " << stressChange[i] << endl;
        }
        StrainType::mult(stressChange[i], kFactor);
    }

    // integrate and compute force
    for(unsigned i=0; i<dx.size(); i++)
    {
        StrainType::addMultTranspose(df[i], dx[i], stressChange[i], this->sampleInteg[i], &rotation[i]);
        if( this->f_printLog.getValue() )
        {
            cerr<<"CorotationalForceField<DataTypes>::addDForce, stress deformation gradient change after accumulating "<< kFactor<<"* df = " << df[i] << endl;
        }
    }
}

}
}
} // namespace sofa


