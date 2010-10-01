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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALFIXEDCONSTRAINT_H

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/component/topology/PointSubset.h>
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
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef topology::PointSubset SetIndex;
    typedef helper::vector<unsigned int> SetIndexArray;


protected:
    PartialFixedConstraintInternalData<DataTypes> data;
    friend class PartialFixedConstraintInternalData<DataTypes>;

public:
    Data<SetIndex> f_indices;
    Data<bool> f_fixAll;
    Data<double> _drawSize;
    enum { NumDimensions = Deriv::total_size };
    typedef sofa::helper::fixed_array<bool,NumDimensions> Vec6Bool;
    Data<Vec6Bool> fixedDirections;  ///< Defines the directions in which the particles are fixed: true (or 1) for fixed, false (or 0) for free.

    PartialFixedConstraint();

    virtual ~PartialFixedConstraint();

    void clearConstraints();
    void addConstraint(unsigned int index);
    void removeConstraint(unsigned int index);

    // -- Constraint interface
    void init();
    template <class DataDeriv>
    void projectResponseT(DataDeriv& dx);

    void projectResponse(VecDeriv& dx);
    void projectResponse(MatrixDerivRowType& dx);
    void projectVelocity(VecDeriv& /*dx*/); ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& /*x*/) {} ///< project x to constrained space (x models a position)

    void applyConstraint(defaulttype::BaseMatrix *mat, unsigned int &offset);
    void applyConstraint(defaulttype::BaseVector *vect, unsigned int &offset);

    // Handle topological changes
    virtual void handleTopologyChange();

    virtual void draw();

    /// this constraint is holonomic
    bool isHolonomic() {return true;}

    bool fixAllDOFs() const { return f_fixAll.getValue(); }

protected:

    sofa::core::topology::BaseMeshTopology* topology;

    // Define TestNewPointFunction
    static bool FCTestNewPointFunction(int, void*, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& );

    // Define RemovalFunction
    static void FCRemovalFunction ( int , void*);
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Vec3dTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Vec2dTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Vec1dTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Vec6dTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Vec3fTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Vec2fTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Vec1fTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Vec6fTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialFixedConstraint<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa


#endif
