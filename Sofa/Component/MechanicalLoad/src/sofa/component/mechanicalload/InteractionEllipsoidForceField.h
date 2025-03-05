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

#include <sofa/core/behavior/MixedInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/type/RGBAColor.h>

namespace sofa::component::mechanicalload
{

/// This class can be overridden if needed for additional storage within template specializations.
template<class DataTypes1, class DataTypes2>
class InteractionEllipsoidForceFieldInternalData
{
public:
};

template<typename TDataTypes1, typename TDataTypes2>
class InteractionEllipsoidForceField : public core::behavior::MixedInteractionForceField<TDataTypes1, TDataTypes2>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(InteractionEllipsoidForceField, TDataTypes1, TDataTypes2), SOFA_TEMPLATE2(core::behavior::MixedInteractionForceField, TDataTypes1, TDataTypes2));

    typedef core::behavior::MixedInteractionForceField<TDataTypes1, TDataTypes2> Inherit;
    typedef TDataTypes1 DataTypes1;
    typedef typename DataTypes1::VecCoord VecCoord1;
    typedef typename DataTypes1::VecDeriv VecDeriv1;
    typedef typename DataTypes1::Coord    Coord1;
    typedef typename DataTypes1::Deriv    Deriv1;
    typedef typename DataTypes1::Real     Real1;
    typedef TDataTypes2 DataTypes2;
    typedef typename DataTypes2::VecCoord VecCoord2;
    typedef typename DataTypes2::VecDeriv VecDeriv2;
    typedef typename DataTypes2::Coord    Coord2;
    typedef typename DataTypes2::Deriv    Deriv2;
    typedef typename DataTypes2::Real     Real2;

    typedef core::objectmodel::Data<VecCoord1>    DataVecCoord1;
    typedef core::objectmodel::Data<VecDeriv1>    DataVecDeriv1;
    typedef core::objectmodel::Data<VecCoord2>    DataVecCoord2;
    typedef core::objectmodel::Data<VecDeriv2>    DataVecDeriv2;



    enum { N=DataTypes1::spatial_dimensions };
    typedef type::Mat<N,N,Real1> Mat;
protected:
    class Contact
    {
    public:
        int index;
        Deriv1 pos,force;
        sofa::type::Vec3 bras_levier;
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

    Data<sofa::type::vector<Contact> > contacts; ///< Contacts

    InteractionEllipsoidForceFieldInternalData<DataTypes1, DataTypes2> data;

    bool calcF(const Coord1& p1, const Deriv1 &v1, Deriv1& f1,Mat& dfdx);
    void initCalcF();

public:
    Data<VecCoord1> center; ///< ellipsoid center
    Data<VecCoord1> vradius; ///< ellipsoid radius
    Data<Real1> stiffness; ///< force stiffness (positive to repulse outward, negative inward)
    Data<Real1> damping; ///< force damping
    Data<sofa::type::RGBAColor> color; ///< ellipsoid color. (default=[0.0,0.5,1.0,1.0])
    Data<bool> bDraw; ///< enable/disable drawing of the ellipsoid
    Data<int> object2_dof_index; ///< Dof index of object 2 where the forcefield is attached
    Data<bool> object2_forces; ///< enable/disable propagation of forces to object 2
    Data<bool> object2_invert; ///< inverse transform from object 2 (use when object 1 is in local coordinates within a frame defined by object 2)

protected:
    InteractionEllipsoidForceField()
        : contacts(initData(&contacts,"contacts", "Contacts"))
        , center(initData(&center, "center", "ellipsoid center"))
        , vradius(initData(&vradius, "vradius", "ellipsoid radius"))
        , stiffness(initData(&stiffness, (Real1)500, "stiffness", "force stiffness (positive to repulse outward, negative inward)"))
        , damping(initData(&damping, (Real1)5, "damping", "force damping"))
        , color(initData(&color, sofa::type::RGBAColor(0.0f,0.5f,1.0f,1.0f), "color", "ellipsoid color. (default=[0.0,0.5,1.0,1.0])"))
        , bDraw(initData(&bDraw, true, "draw", "enable/disable drawing of the ellipsoid"))
        , object2_dof_index(initData(&object2_dof_index, (int)0, "object2_dof_index", "Dof index of object 2 where the forcefield is attached"))
        , object2_forces(initData(&object2_forces, true, "object2_forces", "enable/disable propagation of forces to object 2"))
        , object2_invert(initData(&object2_invert, false, "object2_invert", "inverse transform from object 2 (use when object 1 is in local coordinates within a frame defined by object 2)"))
    {
    }
public:
    void setStiffness(Real1 stiff)
    {
        stiffness.setValue( stiff );
    }

    void setDamping(Real1 damp)
    {
        damping.setValue( damp );
    }

    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv1& f1, DataVecDeriv2& f2, const DataVecCoord1& x1, const DataVecCoord2& x2, const DataVecDeriv1& v1, const DataVecDeriv2& v2) override;

    virtual void addForce2(DataVecDeriv1& f1, DataVecDeriv2& f2, const DataVecCoord1& p1, const DataVecCoord2& p2, const DataVecDeriv1& v1, const DataVecDeriv2& v2);

    void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv1& df1, DataVecDeriv2& df2, const DataVecDeriv1& dx1, const DataVecDeriv2& dx2) override;

    SReal getPotentialEnergy(const sofa::core::MechanicalParams* mparams, const DataVecCoord1& x1, const DataVecCoord2& x2)const override;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    void init() override;
    void reinit() override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:
    struct TempVars
    {
        unsigned int nelems;
        VecCoord1 vcenter; // center in the local frame ;
        VecCoord1 vr;
        Real1 stiffness;
        Real1 stiffabs;
        Real1 damping;
        VecCoord1 vinv_r2;
        Coord2 pos6D;
    } vars;
};

#if !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_INTERACTIONELLIPSOIDFORCEFIELD_CPP)
extern template class InteractionEllipsoidForceField<defaulttype::Vec3Types, defaulttype::Rigid3Types>;


#endif

} // namespace sofa::component::mechanicalload
