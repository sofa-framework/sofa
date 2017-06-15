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
#ifndef SOFA_COMPONENT_FORCEFIELD_VACCUMSPHEREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_VACCUMSPHEREFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Event.h>

#include <sofa/defaulttype/RGBAColor.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class VaccumSphereForceFieldInternalData
{
public:
};

template<class DataTypes>
class VaccumSphereForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(VaccumSphereForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

protected:
    class Contact
    {
    public:
        int index;
        Coord normal;
        Real fact;
        Contact( int index=0, Coord normal=Coord(),Real fact=Real(0))
            : index(index),normal(normal),fact(fact)
        {
        }

        inline friend std::istream& operator >> ( std::istream& in, Contact& c )
        {
            in>>c.index>>c.normal>>c.fact;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Contact& c )
        {
            out << c.index << " " << c.normal << " " << c.fact ;
            return out;
        }

    };

    Data<sofa::helper::vector<Contact> > contacts;

    core::behavior::MechanicalState<DataTypes> * centerDOF;

    VaccumSphereForceFieldInternalData<DataTypes> data;

public:

    Data<Coord> sphereCenter;
    Data<Real> sphereRadius;
    Data<Real> stiffness;
    Data<Real> damping;
    Data<defaulttype::RGBAColor> color;
    Data<bool> bDraw;
    Data<std::string> centerState;
    Data < bool > active;
    Data < char > keyEvent;
    Data < Real > filter;
protected:
    VaccumSphereForceField()
        : contacts(initData(&contacts,"contacts", "Contacts"))
        , centerDOF(NULL)
        , sphereCenter(initData(&sphereCenter, "center", "sphere center"))
        , sphereRadius(initData(&sphereRadius, (Real)1, "radius", "sphere radius"))
        , stiffness(initData(&stiffness, (Real)500, "stiffness", "force stiffness"))
        , damping(initData(&damping, (Real)5, "damping", "force damping"))
        , color(initData(&color, defaulttype::RGBAColor(0.0f,0.0f,1.0f,1.0f), "color", "sphere color. (default=[0,0,1,1])"))
        , bDraw(initData(&bDraw, true, "draw", "enable/disable drawing of the sphere"))
        , centerState(initData(&centerState, "centerState", "path to the MechanicalState controlling the center point"))
        , active( initData(&active, false, "active", "Activate this object.\nNote that this can be dynamically controlled by using a key") )
        , keyEvent( initData(&keyEvent, '1', "key", "key to press to activate this object until the key is released") )
        , filter(initData(&filter, (Real)0, "filter", "filter"))
    {
    }
protected:
    void setSphere(const Coord& center, Real radius)
    {
        sphereCenter.setValue( center );
        sphereRadius.setValue( radius );
    }

    void setStiffness(Real stiff)
    {
        stiffness.setValue( stiff );
    }

    void setDamping(Real damp)
    {
        damping.setValue( damp );
    }

    virtual void init();

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    virtual void updateStiffness( const VecCoord& x );

    virtual void handleEvent(sofa::core::objectmodel::Event* event);

    void draw(const core::visual::VisualParams* vparams);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_VACCUMSPHEREFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API VaccumSphereForceField<sofa::defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API VaccumSphereForceField<sofa::defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API VaccumSphereForceField<sofa::defaulttype::Vec1dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API VaccumSphereForceField<sofa::defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API VaccumSphereForceField<sofa::defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API VaccumSphereForceField<sofa::defaulttype::Vec1fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_VACCUMSPHEREFORCEFIELD_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_VACCUMSPHEREFORCEFIELD_H
