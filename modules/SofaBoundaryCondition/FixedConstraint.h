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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDCONSTRAINT_H
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
#include <set>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class FixedConstraintInternalData
{

};


/** Maintain a constant velocity.
 * If the particle is initially fixed then it is attached to its initial position.
 * Otherwise it keeps on drifting.
 * For maintaining particles fixed in any case, @sa ProjectToPointConstraint
*/
template <class DataTypes>
class FixedConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;
    typedef sofa::core::topology::Point Point;
    typedef sofa::defaulttype::Vector3 Vector3;
protected:
    FixedConstraint();

    virtual ~FixedConstraint();

public:
    SetIndex d_indices;
    Data<bool> d_fixAll;
    Data<bool> d_showObject;
    Data<SReal> d_drawSize;
    Data<bool> d_projectVelocity;


protected:
    FixedConstraintInternalData<DataTypes>* data;
    friend class FixedConstraintInternalData<DataTypes>;


public:
    void clearConstraints();
    void addConstraint(unsigned int index);
    void removeConstraint(unsigned int index);

    // -- Constraint interface
    virtual void init() override;
    virtual void reinit() override;

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;


    virtual void applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    virtual void applyConstraint(const core::MechanicalParams* mparams, defaulttype::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    /** Project the the given matrix (Experimental API).
      See doc in base parent class
      */
    virtual void projectMatrix( sofa::defaulttype::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;


    virtual void draw(const core::visual::VisualParams* vparams) override;

    bool fixAllDOFs() const { return d_fixAll.getValue(); }

    class FCPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >
    {
    public:
        typedef typename FixedConstraint<DataTypes>::SetIndexArray SetIndexArray;

        FCPointHandler(FixedConstraint<DataTypes>* _fc, sofa::component::topology::PointSubsetData<SetIndexArray>* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >(_data), fc(_fc) {}


        void applyDestroyFunction(unsigned int /*index*/, value_type& /*T*/);


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        FixedConstraint<DataTypes> *fc;
    };

protected :
    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;

    /// Handler for subset Data
    FCPointHandler* pointHandler;

};

// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void FixedConstraint<defaulttype::Rigid3dTypes >::draw(const core::visual::VisualParams* vparams);
template <>
void FixedConstraint<defaulttype::Rigid2dTypes >::draw(const core::visual::VisualParams* vparams);
#endif
#ifndef SOFA_DOUBLE
template <>
void FixedConstraint<defaulttype::Rigid3fTypes >::draw(const core::visual::VisualParams* vparams);
template <>
void FixedConstraint<defaulttype::Rigid2fTypes >::draw(const core::visual::VisualParams* vparams);
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Vec1dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Vec6dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Vec1fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Vec6fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API FixedConstraint<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif
