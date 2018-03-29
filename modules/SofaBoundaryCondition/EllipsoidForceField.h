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
#ifndef SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_H
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
class EllipsoidForceFieldInternalData
{
public:
};

template<class DataTypes>
class EllipsoidForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(EllipsoidForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;
    typedef Data<VecCoord>                  DataVecCoord;
    typedef Data<VecDeriv>                  DataVecDeriv;

    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;

protected:
    class Contact
    {
    public:
        int index;
        Mat m;
        Contact( int index=0, const Mat& m=Mat())
            : index(index), m(m)
        {
        }

        inline friend std::istream& operator >> ( std::istream& in, Contact& c )
        {
            in>>c.index>>c.m;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Contact& c )
        {
            out << c.index << " " << c.m ;
            return out;
        }

    };

    Data<sofa::helper::vector<Contact> > contacts; ///< Contacts

    EllipsoidForceFieldInternalData<DataTypes> data;

public:

    Data<Coord> center; ///< ellipsoid center
    Data<Coord> vradius; ///< ellipsoid radius
    Data<Real> stiffness; ///< force stiffness (positive to repulse outward, negative inward)
    Data<Real> damping; ///< force damping
    Data<defaulttype::RGBAColor> color; ///< ellipsoid color. (default=0,0.5,1.0,1.0)
    Data<bool> bDraw; ///< enable/disable drawing of the ellipsoid
protected:
    EllipsoidForceField()
        : contacts(initData(&contacts,"contacts", "Contacts"))
        , center(initData(&center, "center", "ellipsoid center"))
        , vradius(initData(&vradius, "vradius", "ellipsoid radius"))
        , stiffness(initData(&stiffness, (Real)500, "stiffness", "force stiffness (positive to repulse outward, negative inward)"))
        , damping(initData(&damping, (Real)5, "damping", "force damping"))
        , color(initData(&color, defaulttype::RGBAColor(0.0f,0.5f,1.0f,1.0f), "color", "ellipsoid color. (default=0,0.5,1.0,1.0)"))
        , bDraw(initData(&bDraw, true, "draw", "enable/disable drawing of the ellipsoid"))
    {
    }
public:
    void setStiffness(Real stiff)
    {
        stiffness.setValue( stiff );
    }

    void setDamping(Real damp)
    {
        damping.setValue( damp );
    }

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;
    ///SOFA_DEPRECATED_ForceField <<<virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;
    ///SOFA_DEPRECATED_ForceField <<<virtual void addDForce (VecDeriv& df, const VecDeriv& dx, SReal kFactor, SReal bFactor);

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    void draw(const core::visual::VisualParams* vparams) override;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<sofa::defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<sofa::defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<sofa::defaulttype::Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<sofa::defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<sofa::defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<sofa::defaulttype::Vec1fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_H
