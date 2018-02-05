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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALLINEARMOVEMENTCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALLINEARMOVEMENTCONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/TopologySubsetData.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <type_traits>
#include <set>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{


template<class DataTypes>
class PartialLinearMovementConstraintInternalData
{
};

/** impose a motion to given DOFs (translation and rotation) in some directions only.
  The moved and free directioons are the same for all the particles, defined  in the movedDirections attribute.
    The motion between 2 key times is linearly interpolated
*/
template <class TDataTypes>
class PartialLinearMovementConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PartialLinearMovementConstraint,TDataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, TDataTypes));

    typedef TDataTypes DataTypes;
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

protected:
    PartialLinearMovementConstraintInternalData<DataTypes> *data;
    friend class PartialLinearMovementConstraintInternalData<DataTypes>;

public :
    /// indices of the DOFs the constraint is applied to
    SetIndex m_indices;
    /// the key frames when the motion is defined by the user
    core::objectmodel::Data<helper::vector<Real> > m_keyTimes;
    /// the motions corresponding to the key frames
    core::objectmodel::Data<VecDeriv > m_keyMovements;

    /// attributes to precise display
    /// if showMovement is true we display the expected movement
    /// otherwise we show which are the fixed dofs
    core::objectmodel::Data< bool > showMovement;

    /// the key times surrounding the current simulation time (for interpolation)
    Real prevT, nextT;
    ///the motions corresponding to the surrouding key times
    Deriv prevM, nextM;
    ///initial constrained DOFs position
    VecCoord x0;

    core::objectmodel::Data<bool> linearMovementBetweenNodesInIndices;
    core::objectmodel::Data<unsigned> mainIndice;
    core::objectmodel::Data<unsigned> minDepIndice;
    core::objectmodel::Data<unsigned> maxDepIndice;
    core::objectmodel::Data<helper::vector<Real> > m_imposedDisplacmentOnMacroNodes; ///< imposed displacement at  u1 u2 u3 u4 for 2d case
    ///< and u1 u2 u3 u4 u5 u6 u7 u8 for 3d case
    Data<Real> X0,Y0,Z0;

    enum { NumDimensions = Deriv::total_size };
    typedef sofa::helper::fixed_array<bool,NumDimensions> VecBool;
    core::objectmodel::Data<VecBool> movedDirections;  ///< Defines the directions in which the particles are moved: true (or 1) for fixed, false (or 0) for free.
protected:
    PartialLinearMovementConstraint();

    virtual ~PartialLinearMovementConstraint();
public:
    ///methods to add/remove some indices, keyTimes, keyMovement
    void clearIndices();
    void addIndex(unsigned int index);
    void removeIndex(unsigned int index);
    void clearKeyMovements();
    /**add a new key movement
    @param time : the simulation time you want to set a movement (in sec)
    @param movement : the corresponding motion
    for instance, addKeyMovement(1.0, Deriv(5,0,0) ) will set a translation of 5 in x direction a time 1.0s
    **/
    void addKeyMovement(Real time, Deriv movement);


    /// -- Constraint interface
    void init() override;
    void reset() override;

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData) override;

    virtual void applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    virtual void applyConstraint(const core::MechanicalParams* mparams, defaulttype::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    virtual void draw(const core::visual::VisualParams*) override;

    class FCPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >
    {
    public:
        typedef typename PartialLinearMovementConstraint<DataTypes>::SetIndexArray SetIndexArray;

        FCPointHandler(PartialLinearMovementConstraint<DataTypes>* _lc, sofa::component::topology::PointSubsetData<SetIndexArray>* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >(_data), lc(_lc) {}



        void applyDestroyFunction(unsigned int /*index*/, value_type& /*T*/);


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        PartialLinearMovementConstraint<DataTypes> *lc;
    };

protected:
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams, DataDeriv& dx);

    template <class MyCoord>
    void interpolatePosition(Real cT, typename std::enable_if<!std::is_same<MyCoord, sofa::defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x);
    template <class MyCoord>
    void interpolatePosition(Real cT, typename std::enable_if<std::is_same<MyCoord, sofa::defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x);

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;

private:

    /// to keep the time corresponding to the key times
    Real currentTime;

    /// to know if we found the key times
    bool finished;

    /// find previous and next time keys
    void findKeyTimes();

    /// Handler for subset Data
    FCPointHandler* pointHandler;
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALLINEARMOVEMENTCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Vec1dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Vec6dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Vec1fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Vec6fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PartialLinearMovementConstraint<defaulttype::Rigid3fTypes>;
#endif
#endif


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif

