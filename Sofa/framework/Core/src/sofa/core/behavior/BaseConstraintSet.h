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


#include <sofa/core/config.h>
#include <sofa/linearalgebra/BaseVector.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/MultiVecId.h>

namespace sofa::core::behavior
{

class SOFA_CORE_API BaseConstraintSet : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseConstraintSet, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseConstraintSet)

protected:
    BaseConstraintSet()
        : group(initData(&group, 0, "group", "ID of the group containing this constraint. This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle."))
        , m_constraintIndex(initData(&m_constraintIndex, 0u, "constraintIndex", "Constraint index (first index in the right hand term resolution vector)"))
    {
    }

    ~BaseConstraintSet() override { }

private:
    BaseConstraintSet(const BaseConstraintSet& n) = delete;
    BaseConstraintSet& operator=(const BaseConstraintSet& n) = delete;

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
    virtual void getConstraintViolation(const ConstraintParams* cParams, linearalgebra::BaseVector *v) {
        getConstraintViolation(cParams,v,m_cId);
    }

    /// Construct the Constraint violations vector
    ///
    /// \param v is the result vector that contains the whole constraints violations
    /// \param cIndex is the index of the next constraint equation
    /// \param cParams defines the state vectors to use for positions and velocities. Also defines the order of the constraint (POS, VEL, ACC)
    virtual void getConstraintViolation(const ConstraintParams* /*cParams*/, linearalgebra::BaseVector * /*v*/, unsigned int /*cIndex*/) {
        dmsg_error() << "getConstraintViolation(const ConstraintParams* cParams, linearalgebra::BaseVector *v, const unsigned int cIndex) is not implemented while it should";
    }

protected:

    Data< int > group; ///< ID of the group containing this constraint. This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle.
public:
    Data< unsigned int > m_constraintIndex; ///< Constraint index (first index in the right hand term resolution vector)

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;
    unsigned m_cId;

};

} // namespace sofa::core::behavior
