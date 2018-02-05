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
//
// C++ Interface: ConicalForceField
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_FORCEFIELD_CONICALFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_CONICALFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>

#include <sofa/defaulttype/RGBAColor.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class ConicalForceFieldInternalData
{
public:
};

template<class DataTypes>
class ConicalForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ConicalForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;

    typedef Data<VecCoord>                  DataVecCoord;
    typedef Data<VecDeriv>                  DataVecDeriv;

protected:
    class Contact
    {
    public:
        int index;
        Coord normal;
        Coord pos;
        Contact( int index=0, Coord normal=Coord(),Coord pos=Coord())
            : index(index),normal(normal),pos(pos)
        {
        }

        inline friend std::istream& operator >> ( std::istream& in, Contact& c )
        {
            in>>c.index>>c.normal>>c.pos;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Contact& c )
        {
            out << c.index << " " << c.normal << " " << c.pos ;
            return out;
        }

    };

    Data<sofa::helper::vector<Contact> > contacts;

    ConicalForceFieldInternalData<DataTypes> data;

public:

    Data<Coord> coneCenter;
    Data<Coord> coneHeight;
    Data<Real> coneAngle;

    Data<Real> stiffness;
    Data<Real> damping;
    Data<defaulttype::RGBAColor> color;
    Data<bool> bDraw;
protected:
    ConicalForceField()
        : coneCenter(initData(&coneCenter, "coneCenter", "cone center"))
        , coneHeight(initData(&coneHeight, "coneHeight", "cone height"))
        , coneAngle(initData(&coneAngle, (Real)10, "coneAngle", "cone angle"))

        , stiffness(initData(&stiffness, (Real)500, "stiffness", "force stiffness"))
        , damping(initData(&damping, (Real)5, "damping", "force damping"))
        , color(initData(&color, defaulttype::RGBAColor(0.0f,0.0f,1.0f,1.0f), "color", "cone color. (default=0.0,0.0,0.0,1.0,1.0)"))
        , bDraw(initData(&bDraw, true, "draw", "enable/disable drawing of the cone"))
    {
    }
public:
    void setCone(const Coord& center, Coord height, Real angle)
    {
        coneCenter.setValue( center );
        coneHeight.setValue( height );
        coneAngle.setValue( angle );
    }

    void setStiffness(Real stiff)
    {
        stiffness.setValue( stiff );
    }

    void setDamping(Real damp)
    {
        damping.setValue( damp );
    }

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    virtual void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    virtual void updateStiffness( const VecCoord& x );

    virtual bool isIn(Coord p);

    void draw(const core::visual::VisualParams* vparams) override;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_CONICALFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API ConicalForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API ConicalForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CONICALFORCEFIELD_H
