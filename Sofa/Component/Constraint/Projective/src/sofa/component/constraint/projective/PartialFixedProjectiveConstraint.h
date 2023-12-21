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
#include <sofa/component/constraint/projective/config.h>

#include <sofa/component/constraint/projective/FixedProjectiveConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/vector.h>
#include <sofa/core/topology/TopologySubsetIndices.h>

namespace sofa::component::constraint::projective
{

/**
 * Attach given particles to their initial positions, in some directions only.
 * The fixed and free directioons are the same for all the particles, defined  in the fixedDirections attribute.
 **/
template <class DataTypes>
class PartialFixedProjectiveConstraint : public FixedProjectiveConstraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PartialFixedProjectiveConstraint,DataTypes),SOFA_TEMPLATE(FixedProjectiveConstraint, DataTypes));

    typedef FixedProjectiveConstraint<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef type::vector<sofa::Index> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;

public:
    enum { NumDimensions = Deriv::total_size };
    typedef sofa::type::fixed_array<bool,NumDimensions> VecBool;
    Data<VecBool> d_fixedDirections;  ///< Defines the directions in which the particles are fixed: true (or 1) for fixed, false (or 0) for free.
    Data<bool> d_projectVelocity; ///< activate project velocity to set velocity to zero

protected:
    PartialFixedProjectiveConstraint();
    virtual ~PartialFixedProjectiveConstraint();

public:

    // -- Constraint interface
    void reinit() override;

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    void applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    void projectMatrix( sofa::linearalgebra::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;

    void applyConstraint(sofa::core::behavior::ZeroDirichletCondition* matrix) override;

protected:
    template <class DataDeriv>
    void projectResponseT(DataDeriv& dx,
        const std::function<void(DataDeriv&, const unsigned int, const VecBool&)>& clear);
};

#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDPROJECTIVECONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PartialFixedProjectiveConstraint<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PartialFixedProjectiveConstraint<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PartialFixedProjectiveConstraint<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PartialFixedProjectiveConstraint<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PartialFixedProjectiveConstraint<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PartialFixedProjectiveConstraint<defaulttype::Rigid2Types>;
#endif

} // namespace sofa::component::constraint::projective
