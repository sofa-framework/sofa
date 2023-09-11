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
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::mechanicalload
{

/// Apply constant forces to given degrees of freedom.
template<class DataTypes>
class ConstantForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ConstantForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef type::vector<unsigned int> VecIndex;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::TopologySubsetIndices SetIndex;

    /// indices of the points the force applies to
    SetIndex d_indices;

    /// Concerned DOFs indices are numbered from the end of the MState DOFs vector
    Data< bool > d_indexFromEnd;

    /// Per-point forces.
    Data< VecDeriv > d_forces;

    /// Force applied at each point, if per-point forces are not specified
    SOFA_ATTRIBUTE_DISABLED__CONSTANTFF_FORCE_DATA()
    sofa::core::objectmodel::lifecycle::RemovedData d_force{this, "v23.12", "v24.06", "force", "Replace \"force\" by using the \"forces\" data (providing only one force value) (PR #4019)}"};

    /// Sum of the forces applied at each point, if per-point forces are not specified
    Data< Deriv > d_totalForce;

    /// S for drawing. The sign changes the direction, 0 doesn't draw arrow
    Data< SReal > d_showArrowSize;

    /// display color
    Data< sofa::type::RGBAColor > d_color;

    /// Link to be set to the topology container in the component graph.
    SingleLink<ConstantForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    /// Init function
    void init() override;

    /// Add the forces
    void addForce (const core::MechanicalParams* params, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    /// Constant force has null variation
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df , const DataVecDeriv& d_dx) override;

    /// Constant force has null variation
    void addKToMatrix(sofa::linearalgebra::BaseMatrix *mat, SReal k, unsigned int &offset) override;

    /// Constant force has null variation
    virtual void addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/, SReal /*kFact*/) ;

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;

    SReal getPotentialEnergy(const core::MechanicalParams* params, const DataVecCoord& x) const override;

    void draw(const core::visual::VisualParams* vparams) override;

    /// Set a force to a given particle
    void setForce( unsigned i, const Deriv& force );

    using Inherit::addAlias ;
    using Inherit::addKToMatrix;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final
    {
        // No damping in this ForceField
    }


protected:
    ConstantForceField();

    /// Functions updating data
    sofa::core::objectmodel::ComponentState updateFromIndices();
    sofa::core::objectmodel::ComponentState updateFromForcesVector();
    sofa::core::objectmodel::ComponentState updateFromTotalForce();

    /// Functions checking inputs before update
    bool checkForce(const Deriv&  force);
    bool checkForces(const VecDeriv& forces);

    /// Functions computing and updating the constant force vector
    sofa::core::objectmodel::ComponentState computeForceFromSingleForce(const Deriv singleForce);
    sofa::core::objectmodel::ComponentState computeForceFromForcesVector(const VecDeriv &forces);
    sofa::core::objectmodel::ComponentState computeForceFromTotalForce(const Deriv &totalForce);

    /// Save system size for update of indices (doUpdateInternal)
    size_t m_systemSize;

    /// Boolean specifying whether the data totalMass has been initially given
    /// (else forces vector is being used)
    bool m_isTotalForceUsed;
};

template <>
SReal ConstantForceField<defaulttype::Rigid3Types>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const;
template <>
SReal ConstantForceField<defaulttype::Rigid2Types>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const;

#if !defined(SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API ConstantForceField<sofa::defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API ConstantForceField<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API ConstantForceField<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API ConstantForceField<sofa::defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API ConstantForceField<sofa::defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API ConstantForceField<sofa::defaulttype::Rigid2Types>;
#endif

} // namespace sofa::component::mechanicalload
