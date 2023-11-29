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

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Mat.h>
#include <sofa/type/vector.h>
#include <sofa/type/Quat.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <type_traits>
#include <set>

namespace sofa::component::constraint::projective
{

template<class DataTypes>
class AffineMovementProjectiveConstraintInternalData
{
};

/**
    Impose a motion to all the boundary points of a mesh. The motion of the 4 corners are given in the data m_cornerMovements and the movements of the edge points are computed by linear interpolation.
*/
template <class TDataTypes>
class AffineMovementProjectiveConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(AffineMovementProjectiveConstraint,TDataTypes),
               SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, TDataTypes));

    using Index = sofa::Index;
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef type::vector<Index> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;
    typedef type::Quat<SReal> Quat;
    typedef type::Vec3 Vec3;

    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

    static const auto CoordSize = Coord::total_size;
    typedef type::Mat<3,3,Real> RotationMatrix;

protected:
    AffineMovementProjectiveConstraintInternalData<DataTypes> *data;
    friend class AffineMovementProjectiveConstraintInternalData<DataTypes>;

public :
    /// indices of the DOFs of the mesh
    SetIndex m_meshIndices;
     /// indices of the DOFs the constraint is applied to
    SetIndex m_indices;
    /// data begin time when the constraint is applied
    Data <SReal> m_beginConstraintTime;
    /// data end time when the constraint is applied
    Data <SReal> m_endConstraintTime;
    /// Rotation Matrix of affine transformation
    Data<RotationMatrix> m_rotation;
    /// Quaternion of affine transformation (for rigid)
    Data<Quat> m_quaternion;
    /// Translation Matrix of affine transformation
    Data<Vec3> m_translation;
    /// Draw constrained points
    Data <bool> m_drawConstrainedPoints;
    /// initial constrained DOFs position
    VecCoord x0;
    /// final constrained DOFs position
    VecCoord xf;
    /// initial mesh DOFs position
    VecCoord meshPointsX0;
    /// final mesh DOFs position
    VecCoord meshPointsXf;

    /// Link to be set to the topology container in the component graph.
    SingleLink<AffineMovementProjectiveConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    AffineMovementProjectiveConstraint();

    virtual ~AffineMovementProjectiveConstraint();

public:
    //Add or clear constraints
    void clearConstraints();
    void addConstraint(Index index);
    void removeConstraint(Index index);

    /// -- Constraint interface
    void init() override;

    /// Cancel the possible forces
    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    /// Cancel the possible velocities
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    /// Apply the computed movements to the border mesh points between beginConstraintTime and endConstraintTime
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;

    void projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /* cData */) override
    {
        msg_error() << "projectJacobianMatrix not implemented";
    }

    /// Compute the theoretical final positions
    void getFinalPositions (VecCoord& finalPos, DataVecCoord& xData);

    // Implement projectMatrix for assembled solver of compliant
    void projectMatrix( sofa::linearalgebra::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;

    /// Draw the constrained points (= border mesh points)
     void draw(const core::visual::VisualParams* vparams) override;

protected:

    void projectResponseImpl(VecDeriv& dx);

private:

    /// Initialize initial positions
    void initializeInitialPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0);

     /// Initialize final positions
    void initializeFinalPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0 , VecCoord& xf);

    /// Apply affine transform
    void transform(const SetIndexArray & indices, VecCoord& x0 , VecCoord& xf);
};

#if !defined(SOFABOUNDARYCONDITION_AFFINEMOVEMENTPROJECTIVECONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AffineMovementProjectiveConstraint<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API AffineMovementProjectiveConstraint<defaulttype::Rigid3Types>;
#endif //SOFABOUNDARYCONDITION_AFFINEMOVEMENTPROJECTIVECONSTRAINT_CPP

} // namespace sofa::component::constraint::projective
