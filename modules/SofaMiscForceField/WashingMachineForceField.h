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
//
// C++ Interface: WashingMachineForceField
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_FORCEFIELD_WASHINGMACHINEFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_WASHINGMACHINEFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <SofaBoundaryCondition/PlaneForceField.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace forcefield
{

/// A box of 6 PlaneForceField that can rotate
template<class DataTypes>
class WashingMachineForceField : public core::behavior::ForceField<DataTypes>//, public core::VisualModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(WashingMachineForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef PlaneForceField<DataTypes> PlaneForceFieldT;
protected:
    core::objectmodel::Data<Coord> _center; ///< box center
    core::objectmodel::Data<Deriv> _size; ///< box size
    core::objectmodel::Data<Real> _speed; ///< rotation speed
    core::objectmodel::Data<Deriv> _axis; ///< rotation axis

    core::objectmodel::Data<Real> _stiffness; ///< penality force stiffness
    core::objectmodel::Data<Real> _damping; ///< penality force damping

    defaulttype::Vec<6, typename PlaneForceFieldT::SPtr> _planes;

    WashingMachineForceField(core::behavior::MechanicalState<DataTypes>* object=NULL, const std::string& /*name*/="")
        : core::behavior::ForceField<DataTypes>(object)
        , _center(initData(&_center, Coord(0,0,0), "center", "box center"))
        , _size(initData(&_size, Deriv(1,1,1), "size", "box size"))
        , _speed(initData(&_speed, (Real)0.001, "speed", "rotation speed"))
        , _axis(initData(&_axis, Deriv(1,0,0), "axis", "rotation axis"))
        , _stiffness(initData(&_stiffness, (Real)500.0, "stiffness", "penality force stiffness"))
        , _damping(initData(&_damping, (Real)5.0, "damping", "penality force damping"))
    {
        _alreadyInit=false;
    }


    ~WashingMachineForceField()
    {
    }

public:

    virtual void init() override
    {
        _alreadyInit=true;

        Inherit::init();

        for(int i=0; i<6; ++i)
        {
            _planes[i] = sofa::core::objectmodel::New< PlaneForceFieldT >();
            this->addSlave(_planes[i]);
            _planes[i]->setStiffness(_stiffness.getValue());
            _planes[i]->setDamping(_damping.getValue());
            _planes[i]->setMState( this->mstate );
        }

        Deriv diff = _center.getValue() - _size.getValue() * .5;
        Deriv diff2 = - _center.getValue() - _size.getValue() * .5;

        _planes[0]->setPlane( Deriv( 0, 1, 0), diff[1] ); // sud
        _planes[1]->setPlane( Deriv( 0, -1, 0), diff2[1] ); // nord
        _planes[2]->setPlane( Deriv( -1, 0, 0), diff2[0] ); // ouest
        _planes[3]->setPlane( Deriv( 1, 0, 0), diff[0] ); // est
        _planes[4]->setPlane( Deriv( 0, 0, 1), diff[2] ); // derriere
        _planes[5]->setPlane( Deriv( 0, 0, -1), diff2[2] ); //devant

        _planes[0]->setDrawColor( defaulttype::RGBAColor( 0.5f,0.4f,0.4f,1.0f ) );
        _planes[1]->setDrawColor( defaulttype::RGBAColor( 0.4f,0.5f,0.4f,1.0f ) );
        _planes[2]->setDrawColor( defaulttype::RGBAColor( 0.4f,0.4f,0.5f,1.0f ) );
        _planes[3]->setDrawColor( defaulttype::RGBAColor( 0.5f,0.5f,0.4f,1.0f ) );
        _planes[4]->setDrawColor( defaulttype::RGBAColor( 0.5f,0.4f,0.5f,1.0f ) );
        _planes[5]->setDrawColor( defaulttype::RGBAColor( 0.4f,0.5f,0.5f,1.0f ) );
    }

    virtual void reinit() override
    {
        Inherit::reinit();

        for(int i=0; i<6; ++i)
        {
            _planes[i]->setStiffness(_stiffness.getValue());
            _planes[i]->setDamping(_damping.getValue());
            _planes[i]->setMState( this->mstate );
        }

        Deriv diff = _center.getValue() - _size.getValue() * .5;
        Deriv diff2 = - _center.getValue() - _size.getValue() * .5;

        _planes[0]->setPlane( Deriv( 0, 1, 0), diff[1]  ); // sud
        _planes[1]->setPlane( Deriv( 0, -1, 0), diff2[1]  ); // nord
        _planes[2]->setPlane( Deriv( -1, 0, 0), diff2[0]  ); // ouest
        _planes[3]->setPlane( Deriv( 1, 0, 0), diff[0]  ); // est
        _planes[4]->setPlane( Deriv( 0, 0, 1), diff[2]  ); // derriere
        _planes[5]->setPlane( Deriv( 0, 0, -1), diff2[2]  ); //devant

        _planes[0]->setDrawColor( defaulttype::RGBAColor( 0.5f,0.4f,0.4f,1.0f ) );
        _planes[1]->setDrawColor( defaulttype::RGBAColor( 0.4f,0.5f,0.4f,1.0f ) );
        _planes[2]->setDrawColor( defaulttype::RGBAColor( 0.4f,0.4f,0.5f,1.0f ) );
        _planes[3]->setDrawColor( defaulttype::RGBAColor( 0.5f,0.5f,0.4f,1.0f ) );
        _planes[4]->setDrawColor( defaulttype::RGBAColor( 0.5f,0.4f,0.5f,1.0f ) );
        _planes[5]->setDrawColor( defaulttype::RGBAColor( 0.4f,0.5f,0.5f,1.0f ) );

    }

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }
    void draw(const core::visual::VisualParams* vparams) override;
    bool addBBox(SReal* minBBox, SReal* maxBBox);


protected :
    bool _alreadyInit;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_WASHINGMACHINEFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class WashingMachineForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class WashingMachineForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
