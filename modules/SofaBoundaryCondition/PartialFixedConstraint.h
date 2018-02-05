/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/TopologySubsetData.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class PartialFixedConstraintInternalData
{
};

/**
 * Attach given particles to their initial positions, in some directions only.
 * The fixed and free directioons are the same for all the particles, defined  in the fixedDirections attribute.
 **/
template <class DataTypes>
class PartialFixedConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PartialFixedConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

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
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;
    typedef sofa::defaulttype::Vector3 Vector3;

protected:
    PartialFixedConstraintInternalData<DataTypes> data;
    friend class PartialFixedConstraintInternalData<DataTypes>;

public:
    SetIndex d_indices;
    Data<bool> d_fixAll;
    Data<SReal> d_drawSize;
    enum { NumDimensions = Deriv::total_size };
    typedef sofa::helper::fixed_array<bool,NumDimensions> VecBool;
    Data<VecBool> fixedDirections;  ///< Defines the directions in which the particles are fixed: true (or 1) for fixed, false (or 0) for free.
    Data<bool> d_projectVelocity;
protected:
    PartialFixedConstraint();

    virtual ~PartialFixedConstraint();
public:
    void clearConstraints();
    void addConstraint(unsigned int index);
    void removeConstraint(unsigned int index);

    // -- Constraint interface
    void init() override;

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    using core::behavior::ProjectiveConstraintSet<DataTypes>::applyConstraint;
    void applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset);
    void applyConstraint(defaulttype::BaseVector *vect, unsigned int offset);
    virtual void applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;


    virtual void projectMatrix( sofa::defaulttype::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;


    virtual void draw(const core::visual::VisualParams*) override;

    bool fixAllDOFs() const { return d_fixAll.getValue(); }


    class FCPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >
    {
    public:
        typedef typename PartialFixedConstraint<DataTypes>::SetIndexArray SetIndexArray;

        FCPointHandler(PartialFixedConstraint<DataTypes>* _fc, sofa::component::topology::PointSubsetData<SetIndexArray>* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >(_data), fc(_fc) {}



        void applyDestroyFunction(unsigned int /*index*/, value_type& /*T*/);


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        PartialFixedConstraint<DataTypes> *fc;
    };

protected:
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams, DataDeriv& dx);

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;

    /// Handler for subset Data
    FCPointHandler* pointHandler;
};


// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void PartialFixedConstraint<defaulttype::Rigid3dTypes >::draw(const core::visual::VisualParams* vparams);
template <>
void PartialFixedConstraint<defaulttype::Rigid2dTypes >::draw(const core::visual::VisualParams* vparams);
#endif
#ifndef SOFA_DOUBLE
template <>
void PartialFixedConstraint<defaulttype::Rigid3fTypes >::draw(const core::visual::VisualParams* vparams);
template <>
void PartialFixedConstraint<defaulttype::Rigid2fTypes >::draw(const core::visual::VisualParams* vparams);
#endif


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Vec1dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Vec6dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Vec1fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Vec6fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialFixedConstraint<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif
