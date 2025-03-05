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
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::mechanicalload
{

/// This class can be overridden if needed for additional storage within
/// template specializations.
template<class DataTypes>
class PlaneForceFieldInternalData
{
public:
};

/// @class PlaneForceField
/// A plane is cutting the space in two half spaces. This component generate a force preventing the
/// object to cross the plane. The plane is defined by its normal and by the amount of displacement
/// along this normal.
template<class DataTypes>
class PlaneForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PlaneForceField, DataTypes),
               SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));
    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::DPos DPos;

protected:
    sofa::type::vector<unsigned int> m_contacts;

    PlaneForceFieldInternalData<DataTypes> m_data;

public:
    Data<DPos> d_planeNormal; ///< plane normal. (default=[0,1,0])
    Data<Real> d_planeD; ///< plane d coef. (default=0)
    Data<Real> d_stiffness; ///< force stiffness. (default=500)
    Data<Real> d_damping; ///< force damping. (default=5)
    Data<Real> d_maxForce; ///< if non-null , the max force that can be applied to the object. (default=0)

    /// option bilateral : if true, the force field is applied on both side of the plane
    Data<bool> d_bilateral;

    /// optional range of local DOF indices. Any computation involving indices outside of this
    /// range are discarded (useful for parallelization using mesh partitioning)
    Data< type::Vec<2,int> > d_localRange;

    Data<bool>                   d_drawIsEnabled; ///< enable/disable drawing of plane. (default=false)
    Data<sofa::type::RGBAColor> d_drawColor; ///< plane color. (default=[0.0,0.5,0.2,1.0])
    Data<Real>                   d_drawSize; ///< plane display size if draw is enabled. (default=10)

protected:
    PlaneForceField() ;

public:
    void setPlane(const Deriv& normal, Real d);
    void setMState(  core::behavior::MechanicalState<DataTypes>* mstate ) { this->mstate = mstate; }

    void setStiffness(Real stiff) { d_stiffness.setValue( stiff ); }
    Real getStiffness() const { return d_stiffness.getValue(); }

    void setDamping(Real damp){ d_damping.setValue( damp ); }
    Real getDamping() const { return d_damping.getValue(); }

    void setDrawColor(const sofa::type::RGBAColor& newvalue){ d_drawColor.setValue(newvalue); }
    const sofa::type::RGBAColor& getDrawColor() const { return d_drawColor.getValue(); }

    //TODO(dmarchal): do we really need a rotate operation into a plan class ?
    void rotate( Deriv axe, Real angle ); // around the origin (0,0,0)

    /// Inherited from ForceField.
    void init() override;
    void addForce(const core::MechanicalParams* mparams,
                          DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const core::MechanicalParams* mparams,
                           DataVecDeriv& df, const DataVecDeriv& dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/,
                                     const DataVecCoord&  /* x */) const override;
    virtual void updateStiffness( const VecCoord& x );
    void addKToMatrix(const core::MechanicalParams*
                              mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) override;
    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    void draw(const core::visual::VisualParams* vparams) override;
    void drawPlane(const core::visual::VisualParams*, float size=0.0f);
    void computeBBox(const core::ExecParams *, bool onlyVisible=false) override;
};


#if !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API PlaneForceField<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API PlaneForceField<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API PlaneForceField<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API PlaneForceField<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API PlaneForceField<defaulttype::Rigid3Types>;

#endif

} // namespace sofa::component::mechanicalload
