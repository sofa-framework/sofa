/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARMOVEMENTCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARMOVEMENTCONSTRAINT_H
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
class LinearMovementConstraintInternalData
{
};

/** impose a motion to given DOFs (translation and rotation)
	The motion between 2 key times is linearly interpolated
*/
template <class TDataTypes>
class LinearMovementConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(LinearMovementConstraint,TDataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, TDataTypes));

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
    LinearMovementConstraintInternalData<DataTypes> *data;
    friend class LinearMovementConstraintInternalData<DataTypes>;

public :
    /// indices of the DOFs the constraint is applied to
    SetIndex m_indices;
    /// the key frames when the motion is defined by the user
    Data<helper::vector<Real> > m_keyTimes;
    /// the motions corresponding to the key frames
    Data<VecDeriv > m_keyMovements;

    /// attributes to precise display
    /// if showMovement is true we display the expected movement
    /// otherwise we show which are the fixed dofs
    Data< bool > showMovement;

    /// the key times surrounding the current simulation time (for interpolation)
    Real prevT, nextT;
    ///the motions corresponding to the surrouding key times
    Deriv prevM, nextM;
    ///initial constrained DOFs position
    VecCoord x0;
protected:
    LinearMovementConstraint();

    virtual ~LinearMovementConstraint();
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
    void init();
    void reset();

    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData);
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData);
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData);
    void projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData);

    virtual void projectMatrix( sofa::defaulttype::BaseMatrix* /*M*/, unsigned /*offset*/ );

    using core::behavior::ProjectiveConstraintSet<TDataTypes>::applyConstraint;
    void applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset);
    void applyConstraint(defaulttype::BaseVector *vect, unsigned int offset);

    virtual void draw(const core::visual::VisualParams* vparams);

    class FCPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >
    {
    public:
        typedef typename LinearMovementConstraint<DataTypes>::SetIndexArray SetIndexArray;

        FCPointHandler(LinearMovementConstraint<DataTypes>* _lc, sofa::component::topology::PointSubsetData<SetIndexArray>* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >(_data), lc(_lc) {}



        void applyDestroyFunction(unsigned int /*index*/, value_type& /*T*/);


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        LinearMovementConstraint<DataTypes> *lc;
    };

protected:
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams, DataDeriv& dx);

    template <class MyCoord>
    void interpolatePosition(Real cT, typename std::enable_if<!std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x);
    template <class MyCoord>
    void interpolatePosition(Real cT, typename std::enable_if<std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x);

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


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARMOVEMENTCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec1dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec6dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec1fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Vec6fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearMovementConstraint<defaulttype::Rigid3fTypes>;
#endif
#endif


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif

