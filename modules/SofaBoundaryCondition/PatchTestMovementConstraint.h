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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PATCHTESTMOVEMENTCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PATCHTESTMOVEMENTCONSTRAINT_H
#include "config.h"

#include <SofaBaseTopology/TopologySubsetData.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>

#include <type_traits>
#include <set>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

template<class DataTypes>
class PatchTestMovementConstraintInternalData
{
};

/** 
    Impose a motion to all the boundary points of a mesh. The motion of the 4 corners are given in the data m_cornerMovements and the movements of the edge points are computed by linear interpolation. 
*/
template <class TDataTypes>
class PatchTestMovementConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PatchTestMovementConstraint,TDataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;

    static const unsigned int CoordSize = Coord::total_size;

    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

protected:
    PatchTestMovementConstraintInternalData<DataTypes> *data;
    friend class PatchTestMovementConstraintInternalData<DataTypes>;

public :
    /// indices of the DOFs of the mesh
    SetIndex m_meshIndices;
     /// indices of the DOFs the constraint is applied to
    SetIndex m_indices;
    /// data begin time when the constraint is applied
    Data <double> m_beginConstraintTime;
    /// data end time when the constraint is applied
    Data <double> m_endConstraintTime;
    /// coordinates of the DOFs the constraint is applied to
    Data<VecCoord> m_constrainedPoints;
    /// the movements of the corner points (this is the difference between initial and final positions of the 4 corners)
    Data<VecDeriv> m_cornerMovements;
    /// the coordinates of the corner points
    Data<VecCoord> m_cornerPoints;
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
 
protected:
    PatchTestMovementConstraint();

    virtual ~PatchTestMovementConstraint();

public:
    //Add or clear constraints
    void clearConstraints();
    void addConstraint(unsigned int index);
    void removeConstraint(unsigned int index);
   
    /// -- Constraint interface
    void init() override;

    /// Cancel the possible forces
    void projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData) override;
    /// Cancel the possible velocities
    void projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData) override;
    /// Apply the computed movements to the border mesh points between beginConstraintTime and endConstraintTime
    void projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData) override;
    // Implement projectMatrix for assembled solver of compliant
    virtual void projectMatrix( sofa::defaulttype::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;

    void projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /* cData */) override
    {
        serr << "projectJacobianMatrix not implemented" << sendl;
    }

    /// Compute the theoretical final positions
    void getFinalPositions (VecCoord& finalPos, DataVecCoord& xData); 

    /// Draw the constrained points (= border mesh points)
     virtual void draw(const core::visual::VisualParams* vparams) override;

    class FCPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >
    {
    public:
        typedef typename PatchTestMovementConstraint<DataTypes>::SetIndexArray SetIndexArray;

        FCPointHandler(PatchTestMovementConstraint<DataTypes>* _fc, sofa::component::topology::PointSubsetData<SetIndexArray>* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >(_data), fc(_fc) {}

        void applyDestroyFunction(unsigned int /*index*/, value_type& /*T*/);

        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        PatchTestMovementConstraint<DataTypes> *fc;
    };

protected:
  
    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;
    
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams, DataDeriv& dx);

private:

    /// Handler for subset Data
    FCPointHandler* pointHandler;

    /// Find the corners of the grid mesh
    void findCornerPoints();
    
    /// Compute the displacement of each mesh point by linear interpolation with the displacement of corner points
    void computeInterpolatedDisplacement (int pointIndice,const DataVecCoord& xData, Deriv& displacement);

    /// Initialize initial positions
    void initializeInitialPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0);

     /// Initialize final positions
    void initializeFinalPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0 , VecCoord& xf);
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PATCHTESTMOVEMENTCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API PatchTestMovementConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PatchTestMovementConstraint<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API PatchTestMovementConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API PatchTestMovementConstraint<defaulttype::Rigid3fTypes>;
#endif
#endif


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif

