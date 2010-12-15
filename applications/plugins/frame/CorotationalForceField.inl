#include "CorotationalForceField.h"
#include <sofa/helper/PolarDecompose.h>
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

}

template <class DataTypes>
inline typename  CorotationalForceField<DataTypes>::StrainVec  CorotationalForceField<DataTypes>::getVoigtForm(  const Frame& f )
{
    StrainVec s;
    unsigned ei=0;
    for(unsigned j=0; j<DataTypes::material_dimensions; j++)
    {
        for( unsigned k=j; k<DataTypes::material_dimensions; k++ )
        {
            s[ei] = f[j][k];
            ei++;
        }
    }
    return s;
}


template <class DataTypes>
void CorotationalForceField<DataTypes>::addForce(DataVecDeriv& _f , const DataVecCoord& _x , const DataVecDeriv& _v , const core::MechanicalParams* /*mparams*/)
{
    ReadAccessor<DataVecCoord> x(_x);
    ReadAccessor<DataVecDeriv> v(_v);
    WriteAccessor<DataVecDeriv> f(_f);
    r.resize(x.size());
    strain.resize(x.size());
    strainRate.resize(x.size());

    // compute strains and strain rates
    for(unsigned i=0; i<x.size(); i++)
    {
        Frame s;

        // strain, using polar decomposition F = R S
        helper::polar_decomp(x[i].getMaterialFrame(),r[i],s);
        // then put S in vector form
        strain[i] = getVoigtForm(s);

        // strain rate: S_ = R^t F_ where _ denotes time derivative
        s = r[i].multTranspose( v[i].getMaterialFrame());
        // then put it in vector form
        strainRate[i] = getVoigtForm(s);
    }
}

template <class DataTypes>
void CorotationalForceField<DataTypes>::addDForce(DataVecDeriv& /*_df*/ , const DataVecDeriv& /*_dx*/ , const core::MechanicalParams* /*mparams*/)
{
}

}
}
} // namespace sofa


