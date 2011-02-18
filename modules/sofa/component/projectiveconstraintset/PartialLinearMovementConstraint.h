/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALLINEARMOVEMENTCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALLINEARMOVEMENTCONSTRAINT_H

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/helper/vector.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <set>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using helper::vector;
using core::objectmodel::Data;
using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;

/** impose a motion to given DOFs (translation and rotation) in some directions only.
  The moved and free directioons are the same for all the particles, defined  in the movedDirections attribute.
	The motion between 2 key times is linearly interpolated
*/


template <class DataTypes>
class PartialLinearMovementConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PartialLinearMovementConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::Real Real;
    typedef topology::PointSubset SetIndex;
    typedef helper::vector<unsigned int> SetIndexArray;

public :
    /// indices of the DOFs the constraint is applied to
    Data<SetIndex> m_indices;
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

    Data<bool> linearMovementBetweenNodesInIndices;
    Data<unsigned> mainIndice;
    Data<unsigned> minDepIndice;
    Data<unsigned> maxDepIndice;
    Data<helper::vector<Real> > m_imposedDisplacmentOnMacroNodes; ///< imposed displacement at  u1 u2 u3 u4 for 2d case
    ///< and u1 u2 u3 u4 u5 u6 u7 u8 for 3d case
    Data<Real> X0,Y0,Z0;

    enum { NumDimensions = Deriv::total_size };
    typedef sofa::helper::fixed_array<bool,NumDimensions> VecBool;
    Data<VecBool> movedDirections;  ///< Defines the directions in which the particles are moved: true (or 1) for fixed, false (or 0) for free.

    PartialLinearMovementConstraint();

    virtual ~PartialLinearMovementConstraint();

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
    template <class DataDeriv>
    void projectResponseT(DataDeriv& dx);
    void projectResponse(VecDeriv& dx);
    void projectResponse(SparseVecDeriv& dx);

    virtual void projectVelocity(VecDeriv& dx); ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& x); ///< project x to constrained space (x models a position)

    /// Handle topological changes
    virtual void handleTopologyChange();

    virtual void draw();

    /// this constraint is holonomic
    bool isHolonomic() {return true;}

protected:

    sofa::core::topology::BaseMeshTopology* topology;

    /// Define TestNewPointFunction (for topology changes)
    static bool FCTestNewPointFunction(int, void*, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& );

    /// Define RemovalFunction (for topology changes)
    static void FCRemovalFunction ( int , void*);

private:

    /// to keep the time corresponding to the key times
    Real currentTime;

    /// to know if we found the key times
    bool finished;

    /// find previous and next time keys
    void findKeyTimes();
};


#if defined(WIN32) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALLINEARMOVEMENTCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Vec2dTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Vec1dTypes>;
//extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Vec6dTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Vec2fTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Vec1fTypes>;
//extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Vec6fTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<defaulttype::Rigid3fTypes>;
#endif
#endif


} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif

