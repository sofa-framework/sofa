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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_AFFINEMOVEMENTCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_AFFINEMOVEMENTCONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/Quater.h>
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
class AffineMovementConstraintInternalData
{
};

/** 
    Impose a motion to all the boundary points of a mesh. The motion of the 4 corners are given in the data m_cornerMovements and the movements of the edge points are computed by linear interpolation. 
*/
template <class TDataTypes>
class AffineMovementConstraint : public core::behavior::ProjectiveConstraintSet<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(AffineMovementConstraint,TDataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;
    typedef defaulttype::Quat Quat;
    typedef defaulttype::Vector3 Vector3;

    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

    static const unsigned int CoordSize = Coord::total_size;
    typedef defaulttype::Mat<3,3,Real> RotationMatrix;

protected:
    AffineMovementConstraintInternalData<DataTypes> *data;
    friend class AffineMovementConstraintInternalData<DataTypes>;

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
    Data<Vector3> m_translation;
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
    AffineMovementConstraint();

    virtual ~AffineMovementConstraint();

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

    void projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /* cData */) override
    {
        serr << "projectJacobianMatrix not implemented" << sendl;
    }

    /// Compute the theoretical final positions
    void getFinalPositions (VecCoord& finalPos, DataVecCoord& xData); 

    // Implement projectMatrix for assembled solver of compliant
    virtual void projectMatrix( sofa::defaulttype::BaseMatrix* /*M*/, unsigned /*offset*/ ) override;

    /// Draw the constrained points (= border mesh points)
     virtual void draw(const core::visual::VisualParams* vparams) override;

     class FCPointHandler : public component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >
    {
    public:
        typedef typename AffineMovementConstraint<DataTypes>::SetIndexArray SetIndexArray;

        FCPointHandler(AffineMovementConstraint<DataTypes>* _fc, component::topology::PointSubsetData<SetIndexArray>* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >(_data), fc(_fc) {}

        using component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, SetIndexArray >::applyDestroyFunction;
        void applyDestroyFunction(unsigned int /*index*/, core::objectmodel::Data<value_type>& /*T*/);

        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/);
    protected:
        AffineMovementConstraint<DataTypes> *fc;
    };

protected:
  
    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;
    
    template <class DataDeriv>
    void projectResponseT(const core::MechanicalParams* mparams, DataDeriv& dx);

private:

    /// Handler for subset Data
    FCPointHandler* pointHandler;

    /// Initialize initial positions
    void initializeInitialPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0);

     /// Initialize final positions
    void initializeFinalPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0 , VecCoord& xf);

    /// Apply affine transform
    void transform(const SetIndexArray & indices, VecCoord& x0 , VecCoord& xf);
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_AFFINEMOVEMENTCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API AffineMovementConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API AffineMovementConstraint<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API AffineMovementConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API AffineMovementConstraint<defaulttype::Rigid3fTypes>;
#endif
#endif


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif

