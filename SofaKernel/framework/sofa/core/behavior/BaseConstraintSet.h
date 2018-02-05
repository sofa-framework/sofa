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
#ifndef SOFA_CORE_BEHAVIOR_BASECONSTRAINTSET_H
#define SOFA_CORE_BEHAVIOR_BASECONSTRAINTSET_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/core.h>

#include <sofa/defaulttype/BaseVector.h>


namespace sofa
{

namespace core
{

namespace behavior
{

class SOFA_CORE_API BaseConstraintSet : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseConstraintSet, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseConstraintSet)

protected:
    BaseConstraintSet()
        : group(initData(&group, 0, "group", "ID of the group containing this constraint. This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle."))
        , m_constraintIndex(initData(&m_constraintIndex, (unsigned int)0, "constraintIndex", "Constraint index (first index in the right hand term resolution vector)"))
    {
    }

    virtual ~BaseConstraintSet() { }

private:
    BaseConstraintSet(const BaseConstraintSet& n) ;
    BaseConstraintSet& operator=(const BaseConstraintSet& n) ;

public:
    virtual void resetConstraint() {}

    /// Set the id of the constraint (this id is build in the getConstraintViolation function)
    ///
    /// \param cId is Id of the first constraint in the sparse matrix
    virtual void setConstraintId(unsigned cId) {
        m_cId = cId;
    }

    /// Process geometrical data.
    ///
    /// This function is called by the CollisionVisitor, it can be used to process a collision detection specific for the constraint
    virtual void processGeometricalData() {}

    /// Construct the Jacobian Matrix
    ///
    /// \param cId is the result constraint sparse matrix Id
    /// \param cIndex is the index of the next constraint equation: when building the constraint matrix, you have to use this index, and then update it
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex) = 0;

    /// Construct the Constraint violations vector
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *v) {
        getConstraintViolation(cParams,v,m_cId);
    }

    /// Construct the Constraint violations vector
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param cIndex is the index of the next constraint equation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void getConstraintViolation(const ConstraintParams* /*cParams*/, defaulttype::BaseVector * /*v*/, unsigned int /*cIndex*/) {
        dmsg_error() << "getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *v, const unsigned int cIndex) is not implemented while it should";
    }

    /// Useful when the Constraint is applied only on a subset of dofs.
    /// It is automatically called by buildConstraintMatrix
    ///
    /// That way, we can optimize the time spent to transfer quantities through the mechanical mappings.
    /// Every Dofs are inserted by default. The Constraint using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask() = 0;

protected:

    Data< int > group;
public:
    Data< unsigned int > m_constraintIndex; /// Constraint index (first index in the right hand term resolution vector)

    virtual bool insertInNode( objectmodel::BaseNode* node ) override;
    virtual bool removeInNode( objectmodel::BaseNode* node ) override;
    unsigned m_cId;

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
