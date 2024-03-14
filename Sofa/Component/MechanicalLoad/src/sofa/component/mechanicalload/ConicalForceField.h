/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/mechanicalload/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>

#include <sofa/type/RGBAColor.h>

namespace sofa::component::mechanicalload
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

        explicit Contact( int index=0, Coord normal=Coord(),Coord pos=Coord())
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

    Data<sofa::type::vector<Contact> > contacts;

    ConicalForceFieldInternalData<DataTypes> data;

public:

    Data<Coord> coneCenter; ///< cone center
    Data<Coord> coneHeight; ///< cone height
    Data<Real> coneAngle; ///< cone angle

    Data<Real> stiffness; ///< force stiffness
    Data<Real> damping; ///< force damping
    Data<sofa::type::RGBAColor> color; ///< cone color. (default=0.0,0.0,0.0,1.0,1.0)
protected:
    ConicalForceField();

public:
    void setCone(const Coord& center, Coord height, Real angle);
    void setStiffness(Real stiff);
    void setDamping(Real damp);

    void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    virtual void updateStiffness( const VecCoord& x );

    virtual bool isIn(Coord p);

    void draw(const core::visual::VisualParams* vparams) override;
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_CONICALFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API ConicalForceField<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::mechanicalload
