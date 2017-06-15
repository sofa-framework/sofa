/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MASS_MATRIXMASS_INL
#define SOFA_COMPONENT_MASS_MATRIXMASS_INL

#include <SofaMiscForceField/MatrixMass.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <SofaBaseMechanics/AddMToMatrixFunctor.h>

#ifdef SOFA_SUPPORT_MOVING_FRAMES
#include <sofa/core/behavior/InertiaForce.h>
#endif

#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace mass
{

template <class DataTypes, class MassType>
MatrixMass<DataTypes, MassType>::~MatrixMass()
{
}


///////////////////////////////////////////


template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::clear()
{
    VecMass& masses = *f_mass.beginEdit();
    masses.clear();
    f_mass.endEdit();
}


template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::resize(int vsize)
{
    VecMass& masses = *f_mass.beginEdit();
    masses.resize(vsize);
    f_mass.endEdit();
}



///////////////////////////////////////////




// -- Mass interface
template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::addMDx(const core::MechanicalParams*, DataVecDeriv& res, const DataVecDeriv& dx, SReal factor)
{
    const VecMass &masses= *_usedMassMatrices;

    helper::WriteAccessor< DataVecDeriv > _res = res;
    helper::ReadAccessor< DataVecDeriv > _dx = dx;
    if (factor == 1.0)
    {
        for (unsigned int i=0; i<_dx.size(); i++)
        {
            _res[i] += masses[i] * _dx[i];
        }
    }
    else
        for (unsigned int i=0; i<_dx.size(); i++)
        {
            _res[i] += masses[i] * _dx[i] * factor;
        }

}

template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::accFromF(const core::MechanicalParams*, DataVecDeriv& , const DataVecDeriv&)
{
    serr<<"void MatrixMass<DataTypes, MassType>::accFromF(VecDeriv& a, const VecDeriv& f) not yet implemented (need the matrix assembly and inversion)"<<sendl;
}

template <class DataTypes, class MassType>
SReal MatrixMass<DataTypes, MassType>::getKineticEnergy( const core::MechanicalParams*, const DataVecDeriv& ) const
{
    serr<<"void MatrixMass<DataTypes, MassType>::getKineticEnergy not yet implemented"<<sendl;
    return 0;
}

template <class DataTypes, class MassType>
SReal MatrixMass<DataTypes, MassType>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& ) const
{
    serr<<"void MatrixMass<DataTypes, MassType>::getPotentialEnergy not yet implemented"<<sendl;
    return 0;
}

// does nothing by default, need to be specialized in .cpp
template <class DataTypes, class MassType>
sofa::defaulttype::Vector6 MatrixMass<DataTypes, MassType>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& /*vx*/, const DataVecDeriv& /*vv*/  ) const
{
    serr<<"void MatrixMass<DataTypes, MassType>::getMomentum not yet implemented"<<sendl;
    return sofa::defaulttype::Vector6();
}


template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        // gravity
        sofa::defaulttype::Vec3d g ( this->getContext()->getGravity() * (mparams->dt()) );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * (mparams->dt());

        // add weight and inertia force
        for (unsigned int i=0; i<v.size(); i++)
        {
            v[i] += hg;
        }
        d_v.endEdit();
    }
}

#ifdef SOFA_SUPPORT_MOVING_FRAMES
template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return;

    const VecMass &masses= *_usedMassMatrices;
    helper::WriteAccessor< DataVecDeriv > _f = f;

    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);


    helper::ReadAccessor< DataVecCoord > _x = x;
    helper::ReadAccessor< DataVecDeriv > _v = v;
    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = this->getContext()->getPositionInWorld() / vframe;
    aframe = this->getContext()->getPositionInWorld().backProjectVector( aframe );

    // add weight and inertia force
    for (unsigned int i=0; i<masses.size(); i++)
    {
        _f[i] += masses[i]*theGravity + core::behavior::inertiaForce(vframe,aframe,masses[i],_x[i],_v[i]);
    }
}
#else

template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/)
{
    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return;

    const VecMass &masses= *_usedMassMatrices;
    helper::WriteAccessor< DataVecDeriv > _f = f;

    // gravity
    sofa::defaulttype::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);


    // add weight
    for (unsigned int i=0; i<masses.size(); i++)
    {
        _f[i] += masses[i]*theGravity;
    }
}
#endif

template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)

{
    const VecMass &masses= *_usedMassMatrices;
    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());
    for (unsigned int i=0; i<masses.size(); i++)
        calc(r.matrix, masses[i], r.offset + N*i, mFactor);
}


template <class DataTypes, class MassType>
SReal MatrixMass<DataTypes, MassType>::getElementMass(unsigned int /*index*/) const
{
    //NOT IMPLEMENTED YET
    return (sofa::defaulttype::Vector3::value_type)(_defaultValue.getValue());
}

template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const
{
    MassType mElement=f_mass.getValue()[index];
    const int dimension=mElement.getNbLines();

    if ((int)m->rowSize() != dimension || (int)m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>()(m, mElement, 0, 1);

}






//////////////////////////////////



template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::init()
{
    Inherited::init();

    if (f_mass.getValue().empty())
    {
        clear();
        defaultDiagonalMatrices();
        _usingDefaultDiagonalMatrices=true;
    }

    assert( f_mass.getValue().size() == (size_t)this->mstate->getSize() );

    if( this->_lumped.getValue() )
    {
        lumpMatrices();
        _usedMassMatrices = &_lumpedMasses;
    }
    else
    {
        _usedMassMatrices = &f_mass.getValue();
    }
}

template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::reinit()
{
    if( _usingDefaultDiagonalMatrices ) // in case where defaultValue is modified
    {
        clear();
        defaultDiagonalMatrices();
    }

    if( this->_lumped.getValue() ) // in case of _lumped is modified
    {
        lumpMatrices();
        _usedMassMatrices = &_lumpedMasses;
    }
    else
    {
        _usedMassMatrices = &f_mass.getValue();
    }
}

template <class DataTypes, class MassType>
MassType MatrixMass<DataTypes, MassType>::diagonalMass( const Real& m )
{
    MassType diagonalMatrixMass;
    diagonalMatrixMass.identity();
    return diagonalMatrixMass*m;
}

template <class DataTypes, class MassType>
MassType MatrixMass<DataTypes, MassType>::lump( const MassType& m )
{
    MassType lumpedM;
    lumpedM.fill(0);
    for (int i=0; i<m.getNbLines(); i++)
    {
        lumpedM[i][i] = m.line(i).sum();
    }
    return lumpedM;
}

template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::lumpMatrices( )
{
    _lumpedMasses.clear();
    for (unsigned i=0; i<f_mass.getValue().size(); ++i)
    {
        _lumpedMasses.push_back( lump( f_mass.getValue()[i] ) );
    }
}

template <class DataTypes, class MassType>
void MatrixMass<DataTypes, MassType>::defaultDiagonalMatrices( )
{
    VecMass& masses = *f_mass.beginEdit();
    masses.resize(this->mstate->getSize());
    MassType diagonalMatrixMass = diagonalMass( _defaultValue.getValue() );
    for (unsigned i=0; i<masses.size(); ++i)
    {
        masses[i] = diagonalMatrixMass;
    }
    _usingDefaultDiagonalMatrices=true;
    f_mass.endEdit();
}

} // namespace mass

} // namespace component

} // namespace sofa

#endif
