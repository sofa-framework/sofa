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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_H
#include "config.h"

#include <set>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/TopologySubsetData.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{
using sofa::defaulttype::BaseVector;
using sofa::core::MechanicalParams;
using sofa::core::visual::VisualParams;
using sofa::core::topology::BaseMeshTopology;
using sofa::core::behavior::MultiMatrixAccessor;
using sofa::core::behavior::ProjectiveConstraintSet;

/// This class can be overriden if needed for additionnal storage within template specilizations.
template <class DataTypes>
class FixedPlaneConstraintInternalData
{
};

template <class DataTypes>
class FixedPlaneConstraint : public ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedPlaneConstraint,DataTypes),
               SOFA_TEMPLATE(ProjectiveConstraintSet, DataTypes));

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
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef component::topology::PointSubsetData< SetIndexArray > SetIndex;
public:
    /// direction on which the constraint applies
    Data<Coord> d_direction;

    Data<Real> d_dmin; ///< coordinates min of the plane for the vertex selection
    Data<Real> d_dmax; ///< coordinates max of the plane for the vertex selection
    SetIndex   d_indices; ///< the set of vertex indices

    /// inherited from the BaseObject interface
    virtual void init() override;
    virtual void draw(const VisualParams* vparams) override;

    /// -- Constraint interface
    virtual void projectResponse(const MechanicalParams* mparams, DataVecDeriv& resData) override;
    virtual void projectVelocity(const MechanicalParams* mparams, DataVecDeriv& vData) override;
    virtual void projectPosition(const MechanicalParams* mparams, DataVecCoord& xData) override;

    /// Implement projectMatrix for assembled solver of compliant
    virtual void projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned offset) override;
    virtual void projectJacobianMatrix(const MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    /// Implement applyConstraint for direct solvers
    virtual void applyConstraint(const MechanicalParams* mparams,
                                 const MultiMatrixAccessor* matrix) override;

    virtual void applyConstraint(const MechanicalParams* mparams, BaseVector* vector,
                                 const MultiMatrixAccessor* matrix) override;

    void setDirection (Coord dir);
    void selectVerticesAlongPlane();
    void setDminAndDmax(const Real _dmin,const Real _dmax);

    void addConstraint(int index);
    void removeConstraint(int index);

protected:
    FixedPlaneConstraint();
    ~FixedPlaneConstraint();

    FixedPlaneConstraintInternalData<DataTypes> data;
    friend class FixedPlaneConstraintInternalData<DataTypes>;

    /// Forward class declaration, definition is in the .inl
    class FCPointHandler;

    /// Handler for subset Data
    FCPointHandler* m_pointHandler;

    /// whether vertices should be selected from 2 parallel planes
    bool m_selectVerticesFromPlanes;

    /// Pointer to the current topology
    BaseMeshTopology* m_topology;

    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using ProjectiveConstraintSet<DataTypes>::mstate;
    using ProjectiveConstraintSet<DataTypes>::getContext;

    /// These two are implemented depending on the templates
    bool isPointInPlane(Coord p) const ;

    /// These two are implemented depending on the templates
    template<class T>
    void projectResponseImpl(const MechanicalParams* mparams, T& dx) const ;
};

#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_CPP)
#ifdef SOFA_WITH_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec6dTypes>;
#endif /// SOFA_WITH_DOUBLE
#ifdef SOFA_WITH_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedPlaneConstraint<defaulttype::Vec6fTypes>;
#endif /// SOFA_WITH_FLOAT
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
