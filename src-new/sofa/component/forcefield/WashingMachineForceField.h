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

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/forcefield/PlaneForceField.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
class WashingMachineForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef PlaneForceField<DataTypes> PlaneForceField;

protected:
    core::objectmodel::DataField<Coord> _center;
    core::objectmodel::DataField<Deriv> _size;
    core::objectmodel::DataField<Real> _speed;

    core::objectmodel::DataField<Real> _stiffness;
    core::objectmodel::DataField<Real> _damping;

    defaulttype::Vec<6,PlaneForceField*> _planes;

public:
    WashingMachineForceField(core::componentmodel::behavior::MechanicalState<DataTypes>* object=NULL, const std::string& /*name*/="")
        : core::componentmodel::behavior::ForceField<DataTypes>(object)
        , _center(dataField(&_center, Coord(0,0,0), "center", "box center"))
        , _size(dataField(&_size, Deriv(1,1,1), "size", "box size"))
        , _speed(dataField(&_speed, (Real)0.01, "speed", "rotation speed"))
        , _stiffness(dataField(&_stiffness, (Real)500.0, "stiffness", "penality force stiffness"))
        , _damping(dataField(&_damping, (Real)5.0, "damping", "penality force damping"))
    {
    }


    ~WashingMachineForceField()
    {
        for(int i=0; i<6; ++i)
            delete _planes[i];
    }


    virtual void init()
    {
        for(int i=0; i<6; ++i)
        {
            _planes[i] = new PlaneForceField(this->mstate);
            _planes[i]->setStiffness(_stiffness.getValue());
            _planes[i]->setDamping(_damping.getValue());
        }

        Deriv diff = _center.getValue() - _size.getValue() * .5;
        Deriv diff2 = - _center.getValue() - _size.getValue() * .5;

        _planes[0]->setPlane( Deriv( 0, 1, 0), diff[1]  ); // sud
        _planes[1]->setPlane( Deriv( 0, -1, 0), diff2[1]  ); // nord
        _planes[2]->setPlane( Deriv( -1, 0, 0), diff2[0]  ); // ouest
        _planes[3]->setPlane( Deriv( 1, 0, 0), diff[0]  ); // est
        _planes[4]->setPlane( Deriv( 0, 0, 1), diff[2]  ); // derriere
        _planes[5]->setPlane( Deriv( 0, 0, -1), diff2[2]  ); //devant

        _planes[0]->_color.setValue( Coord( 0.5f,0.4f,0.4f ) );
        _planes[1]->_color.setValue( Coord( 0.4f,0.5f,0.4f ) );
        _planes[2]->_color.setValue( Coord( 0.4f,0.4f,0.5f ) );
        _planes[3]->_color.setValue( Coord( 0.5f,0.5f,0.4f ) );
        _planes[4]->_color.setValue( Coord( 0.5f,0.4f,0.5f ) );
        _planes[5]->_color.setValue( Coord( 0.4f,0.5f,0.5f ) );

    }

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);


    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
    bool addBBox(double* minBBox, double* maxBBox);
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
