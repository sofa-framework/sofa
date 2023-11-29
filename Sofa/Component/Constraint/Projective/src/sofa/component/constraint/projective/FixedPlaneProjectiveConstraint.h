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

#include <set>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologySubsetIndices.h>

namespace sofa::component::constraint::projective
{

using sofa::linearalgebra::BaseVector;
using sofa::core::MechanicalParams;
using sofa::core::visual::VisualParams;
using sofa::core::topology::BaseMeshTopology;
using sofa::core::behavior::MultiMatrixAccessor;
using sofa::core::behavior::ProjectiveConstraintSet;

/// This class can be overriden if needed for additionnal storage within template specilizations.
template <class DataTypes>
class FixedPlaneProjectiveConstraintInternalData
{
};

template <class DataTypes>
class FixedPlaneProjectiveConstraint : public ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedPlaneProjectiveConstraint,DataTypes),
               SOFA_TEMPLATE(ProjectiveConstraintSet, DataTypes));

    using Index = sofa::Index;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef type::vector<Index> SetIndexArray;
    typedef core::topology::TopologySubsetIndices SetIndex;
public:
    Data<Coord> d_direction; ///< direction on which the constraint applied
    Data<Real> d_dmin; ///< coordinates min of the plane for the vertex selection
    Data<Real> d_dmax; ///< coordinates max of the plane for the vertex selection
    SetIndex   d_indices; ///< the set of vertex indices

    /// Link to be set to the topology container in the component graph.
    SingleLink<FixedPlaneProjectiveConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;


    /// inherited from the BaseObject interface
    void init() override;
    void draw(const VisualParams* vparams) override;

    /// -- Constraint interface
    void projectResponse(const MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const MechanicalParams* mparams, DataVecCoord& xData) override;

    /// Implement projectMatrix for assembled solver of compliant
    void projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned offset) override;
    void projectJacobianMatrix(const MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    /// Implement applyConstraint for direct solvers
    void applyConstraint(const MechanicalParams* mparams,
                                 const MultiMatrixAccessor* matrix) override;

    void applyConstraint(const MechanicalParams* mparams, BaseVector* vect,
                                 const MultiMatrixAccessor* matrix) override;

    void applyConstraint(sofa::core::behavior::ZeroDirichletCondition* matrix) override;

    void setDirection (Coord dir);
    void selectVerticesAlongPlane();
    void setDminAndDmax(const Real _dmin,const Real _dmax);

    void addConstraint(Index index);
    void removeConstraint(Index index);

protected:
    FixedPlaneProjectiveConstraint();
    ~FixedPlaneProjectiveConstraint();

    FixedPlaneProjectiveConstraintInternalData<DataTypes> data;
    friend class FixedPlaneProjectiveConstraintInternalData<DataTypes>;

    /// whether vertices should be selected from 2 parallel planes
    bool m_selectVerticesFromPlanes {false};

    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using ProjectiveConstraintSet<DataTypes>::mstate;
    using ProjectiveConstraintSet<DataTypes>::getContext;

    bool isPointInPlane(const Coord& p) const ;
};

#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANEPROJECTIVECONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API FixedPlaneProjectiveConstraint<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API FixedPlaneProjectiveConstraint<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API FixedPlaneProjectiveConstraint<defaulttype::Vec6Types>;
#endif

} // namespace sofa::component::constraint::projective
