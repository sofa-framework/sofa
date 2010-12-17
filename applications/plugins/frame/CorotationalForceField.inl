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
            material->lumpMomentsStiffness(point,StrainType::strain_order,moments);
            for(unsigned int j=0; j<moments.size() && j<this->sampleInteg[i].size() ; j++)
            {
                this->sampleInteg[i][j]=moments[j];
            }
        }
        else
        {
            this->sampleInteg[i][0]=1; // default value for the volume when model vertices are used as gauss points
        }
    }


}


template <class DataTypes>
void CorotationalForceField<DataTypes>::addForce(DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v , const core::MechanicalParams* /*mparams*/)
{
    ReadAccessor<DataVecCoord> x(_x);
    ReadAccessor<DataVecDeriv> v(_v);
    WriteAccessor<DataVecDeriv> f(_f);
    ReadAccessor<Data<VecMaterialCoord> > out (sampleData->f_materialPoints);
    stressStrainMatrices.resize(out.size());
    rotation.resize(x.size());
    strain.resize(x.size());
    strainRate.resize(x.size());
    stress.resize(x.size());

    // compute strains and strain rates
    for(unsigned i=0; i<x.size(); i++)
    {
        StrainType::getStrain(x[i], strain[i], rotation[i]);
        StrainType::getStrainRate(v[i], strainRate[i], rotation[i]);
    }
    material->computeStress( stress, &stressStrainMatrices, strain, strainRate, out.ref() );

    // Todo: integrate and compute frame

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
    }

    // Todo: apply stiffness matrix and integration factors, compute frame
}

}
}
} // namespace sofa


