#include "CorotationalForceField.h"
#include <sofa/core/objectmodel/BaseContext.h>
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
    core::objectmodel::BaseContext* context = this->getContext();
    material = context->get<Material>();
    if( material==NULL )
    {
        cerr<<"CorotationalForceField<DataTypes>::init(), material not found"<< endl;
    }
    sampleData = context->get<SampleData>();
    if( sampleData==NULL )
    {
        cerr<<"CorotationalForceField<DataTypes>::init(), material not found"<< endl;
    }

}


template <class DataTypes>
void CorotationalForceField<DataTypes>::addForce(DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v , const core::MechanicalParams* /*mparams*/)
{
    ReadAccessor<DataVecCoord> x(_x);
    ReadAccessor<DataVecDeriv> v(_v);
    WriteAccessor<DataVecDeriv> f(_f);
    rotation.resize(x.size());
    strain.resize(x.size());
    strainRate.resize(x.size());
    stress.resize(x.size());

    // compute strains and strain rates
    for(unsigned i=0; i<x.size(); i++)
    {
        StrainType::getStrain(x[i], strain[i], rotation[i]);
        StrainType::getStrainRate(v[i], strainRate[i], rotation[i]);
        //x[i].getCorotationalStrain( rotation[i], strain[i] );
        //v[i].getCorotationalStrainRate( strainRate[i], rotation[i] );
    }

    // compute stresses
    for(unsigned i=0; i<x.size(); i++)
        for(unsigned j=0; j<x[i].size(); j++)
            material->computeStress(stress[i][j], NULL, strain[i][j], strainRate[i][j] );

// compute strain energy (necessary ??)
//double U=0;
//for(unsigned i=0; i<x.size(); i++)
//	{
//	integVec=StrainType::multTranspose(strain[i] , stress[i] );
//	for(unsigned j=0; j<strainenergy_size; j++) U+= integVec[j] * (sampleData[i].sampleInteg)[j] * 0.5;
//	}

    // integrate force in volume
    //for(unsigned i=0; i<x.size(); i++)
    //	{
    //	for(unsigned dof=0; dof< ; dof++)
    //	}

    // compute stresses integrated over the volumes of the samples
//                material->computeStress(stress,strain,strainRate,sampleData->sampleInteg);

    // convert vector form to matrix form
    //for(unsigned i=0; i<x.size(); i++)
    //{
    //    f[i].setStress(stress[i]);
    //}


}

template <class DataTypes>
void CorotationalForceField<DataTypes>::addDForce(DataVecDeriv& _df , const DataVecDeriv&  _dx , const core::MechanicalParams* /*mparams*/)
{
    ReadAccessor<DataVecCoord> dx(_dx);
    WriteAccessor<DataVecDeriv> df(_df);
    strainChange.resize(dx.size());
    stressChange.resize(dx.size());

    // compute strains changes
    for(unsigned i=0; i<dx.size(); i++)
    {
        StrainType::getStrainRate(dx[i], strainRate[i], rotation[i]);
        //dx[i].getCorotationalStrainRate( strainRate[i], rotation[i] );
    }

    //// compute stress changes integrated over the volumes of the samples
    //material->computeStressChange(stressChange,strainChange,sampleData->sampleIntegVector);

    //// convert vector form to matrix form
    //for(unsigned i=0; i<dx.size(); i++)
    //{
    //    df[i].setStress(stressChange[i]);
    //}
}

}
}
} // namespace sofa


