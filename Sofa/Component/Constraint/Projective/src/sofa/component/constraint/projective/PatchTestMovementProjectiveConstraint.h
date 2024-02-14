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

#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/vector.h>

#include <type_traits>
#include <set>

namespace sofa::component::constraint::projective
{

template<class DataTypes>
class PatchTestMovementProjectiveConstraintInternalData
{
};

/** 
    Impose a motion to all the boundary points of a mesh. The motion of the 4 corners are given in the data d_cornerMovements and the movements of the edge points are computed by linear interpolation.
*/
template <class TDataTypes>
class PatchTestMovementProjectiveConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PatchTestMovementProjectiveConstraint,TDataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, TDataTypes));

    using Index = sofa::Index;
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef type::vector<Index> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;

    static constexpr unsigned int CoordSize = Coord::total_size;

    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

protected:
    PatchTestMovementProjectiveConstraintInternalData<DataTypes> *data;
    friend class PatchTestMovementProjectiveConstraintInternalData<DataTypes>;

public :
    /// indices of the DOFs of the mesh
    SetIndex d_meshIndices;
     /// indices of the DOFs the constraint is applied to
    SetIndex d_indices;
    /// data begin time when the constraint is applied
    Data <double> d_beginConstraintTime;
    /// data end time when the constraint is applied
    Data <double> d_endConstraintTime;
    /// coordinates of the DOFs the constraint is applied to
    Data<VecCoord> d_constrainedPoints;
    /// the movements of the corner points (this is the difference between initial and final positions of the 4 corners)
    Data<VecDeriv> d_cornerMovements;
    /// the coordinates of the corner points
    Data<VecCoord> d_cornerPoints;
    /// Draw constrained points
    Data <bool> d_drawConstrainedPoints;
    /// initial constrained DOFs position
    VecCoord x0;
    /// final constrained DOFs position
    VecCoord xf;
    /// initial mesh DOFs position
    VecCoord meshPointsX0;
    /// final mesh DOFs position
    VecCoord meshPointsXf;
 
    /// Link to be set to the topology container in the component graph.
    SingleLink<PatchTestMovementProjectiveConstraint<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    PatchTestMovementProjectiveConstraint();

    virtual ~PatchTestMovementProjectiveConstraint();

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
    // Implement projectMatrix for assembled solver of compliant
    void projectMatrix( sofa::linearalgebra::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;

    void projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /* cData */) override
    {
        msg_error() <<"projectJacobianMatrix not implemented";
    }

    /// Compute the theoretical final positions
    void getFinalPositions (VecCoord& finalPos, DataVecCoord& xData); 

    /// Draw the constrained points (= border mesh points)
     void draw(const core::visual::VisualParams* vparams) override;

protected:

    void projectResponseImpl(VecDeriv& dx);

private:

    /// Find the corners of the grid mesh
    void findCornerPoints();
    
    /// Compute the displacement of each mesh point by linear interpolation with the displacement of corner points
    void computeInterpolatedDisplacement (int pointIndice,const DataVecCoord& xData, Deriv& displacement);

    /// Initialize initial positions
    void initializeInitialPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0);

     /// Initialize final positions
    void initializeFinalPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0 , VecCoord& xf);
};


#if !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PATCHTESTMOVEMENTPROJECTIVECONSTRAINT_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PatchTestMovementProjectiveConstraint<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_PROJECTIVE_API PatchTestMovementProjectiveConstraint<defaulttype::Rigid3Types>;
#endif


} // namespace sofa::component::constraint::projective
