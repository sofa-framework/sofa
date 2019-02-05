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

    Data<sofa::helper::vector<Contact> > contacts; ///< Contacts
    core::behavior::MechanicalState<DataTypes> * centerDOF;
    VaccumSphereForceFieldInternalData<DataTypes> data;

public:
    Data<Coord> sphereCenter; ///< sphere center
    Data<Real> sphereRadius; ///< sphere radius
    Data<Real> stiffness; ///< force stiffness
    Data<Real> damping; ///< force damping
    Data<defaulttype::RGBAColor> color; ///< sphere color. (default=[0,0,1,1])
    Data<std::string> centerState; ///< path to the MechanicalState controlling the center point
    Data < bool > active; ///< Activate this object. Note that this can be dynamically controlled by using a key
    Data < char > keyEvent; ///< key to press to activate this object until the key is released
    Data < Real > filter; ///< filter

protected:
    VaccumSphereForceField();

    void setSphere(const Coord& center, Real radius);
    void setStiffness(Real stiff);
    void setDamping(Real damp);

    virtual void init() override;
    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override;

    virtual void handleEvent(sofa::core::objectmodel::Event* event) override;
    void draw(const core::visual::VisualParams* vparams) override;

    virtual void updateStiffness( const VecCoord& x );
};

#if  !defined(SOFA_COMPONENT_FORCEFIELD_VACCUMSPHEREFORCEFIELD_CPP)
extern template class SOFA_BOUNDARY_CONDITION_API VaccumSphereForceField<sofa::defaulttype::Vec3Types>;
extern template class SOFA_BOUNDARY_CONDITION_API VaccumSphereForceField<sofa::defaulttype::Vec2Types>;
extern template class SOFA_BOUNDARY_CONDITION_API VaccumSphereForceField<sofa::defaulttype::Vec1Types>;
#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_VACCUMSPHEREFORCEFIELD_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_VACCUMSPHEREFORCEFIELD_H
