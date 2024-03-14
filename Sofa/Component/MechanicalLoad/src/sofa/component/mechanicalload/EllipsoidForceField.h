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

    static constexpr auto N = DataTypes::spatial_dimensions;
    using Mat = type::Mat<N, N, Real>;

protected:
    class Contact
    {
    public:
        int index;
        Mat m;

        explicit Contact( int index=0, const Mat& m=Mat())
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

    Data<sofa::type::vector<Contact> > d_contacts; ///< Vector of contacts
    SOFA_ELLIPSOIDFORCEFIELD_RENAMEDDATA_DISABLED() DeprecatedAndRemoved contacts; ///< Contacts

    EllipsoidForceFieldInternalData<DataTypes> data;

public:

    Data<Coord> d_center; ///< ellipsoid center
    SOFA_ELLIPSOIDFORCEFIELD_RENAMEDDATA_DISABLED() DeprecatedAndRemoved center; ///< ellipsoid center

    Data<Coord> d_vradius; ///< ellipsoid radius
    SOFA_ELLIPSOIDFORCEFIELD_RENAMEDDATA_DISABLED() DeprecatedAndRemoved vradius; ///< ellipsoid radius

    Data<Real> d_stiffness; ///< force stiffness (positive to repulse outward, negative inward)
    SOFA_ELLIPSOIDFORCEFIELD_RENAMEDDATA_DISABLED() DeprecatedAndRemoved stiffness; ///< force stiffness (positive to repulse outward, negative inward)

    Data<Real> d_damping; ///< force damping
    SOFA_ELLIPSOIDFORCEFIELD_RENAMEDDATA_DISABLED() DeprecatedAndRemoved damping; ///< force damping

    Data<sofa::type::RGBAColor> d_color; ///< ellipsoid color. (default=0,0.5,1.0,1.0)
    SOFA_ELLIPSOIDFORCEFIELD_RENAMEDDATA_DISABLED() DeprecatedAndRemoved color; ///< ellipsoid color. (default=0,0.5,1.0,1.0)

protected:
    EllipsoidForceField();
    ~EllipsoidForceField() override;

public:
    void setStiffness(Real stiff);
    void setDamping(Real damp);

    void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) override;

    void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) override;

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override;

    void draw(const core::visual::VisualParams* vparams) override;
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API EllipsoidForceField<sofa::defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API EllipsoidForceField<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API EllipsoidForceField<sofa::defaulttype::Vec1Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_CPP)

} // namespace sofa::component::mechanicalload
