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
#include <sofa/component/solidmechanics/spring/config.h>

#include <sofa/type/RGBAColor.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/type/vector.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/component/solidmechanics/spring/BaseRestShapeSpringsForceField.h>


namespace sofa::component::solidmechanics::spring
{

/**
* @brief This class describes a simple elastic springs ForceField between DOFs positions and rest positions.
*
* Springs are applied to given degrees of freedom between their current positions and their rest shape positions.
* An external MechanicalState reference can also be passed to the ForceField as rest shape position.
*/
template<class DataTypes>
class FixedWeakConstraint : public BaseRestShapeSpringsForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedWeakConstraint, DataTypes), SOFA_TEMPLATE(BaseRestShapeSpringsForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef type::vector< sofa::Index > VecIndex;
    typedef sofa::core::topology::TopologySubsetIndices DataSubsetIndex;
    typedef type::vector< Real >	 VecReal;

    static constexpr sofa::Size spatial_dimensions = Coord::spatial_dimensions;
    static constexpr sofa::Size coord_total_size = Coord::total_size;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::TopologySubsetIndices SetIndex;

    Data<bool> d_fixAll;

protected:
    FixedWeakConstraint();
    virtual bool checkOutOfBoundsIndices() override;

public:
    virtual void init() override;

    /// Add the forces.
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;

    /// Brings ForceField contribution to the global system stiffness matrix.
    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* matrix) override;

    void draw(const core::visual::VisualParams* vparams) override;


};

#if !defined(SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGSFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<sofa::defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<sofa::defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FixedWeakConstraint<sofa::defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::solidmechanics::spring
