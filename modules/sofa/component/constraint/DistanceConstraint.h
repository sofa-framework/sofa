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
#ifndef SOFA_COMPONENT_CONSTRAINT_DISTANCECONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_DISTANCECONSTRAINT_H

#include <sofa/core/componentmodel/behavior/BaseMass.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/core/componentmodel/behavior/LMConstraint.h>
namespace sofa
{

namespace component
{

namespace constraint
{

using helper::vector;
using core::objectmodel::Data;
using namespace sofa::core::objectmodel;

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class DistanceConstraintInternalData
{
};




/** Keep two particules at an initial distance
 */
template <class DataTypes>
class DistanceConstraint :  public core::componentmodel::behavior::LMConstraint<DataTypes,DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecConst VecConst;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef typename sofa::core::componentmodel::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef typename sofa::core::componentmodel::topology::BaseMeshTopology::Edge Edge;


    typedef typename core::componentmodel::behavior::BaseMechanicalState::VecId VecId;
    typedef core::componentmodel::behavior::BaseLMConstraint::ConstOrder ConstOrder;

    using core::componentmodel::behavior::LMConstraint<DataTypes,DataTypes>::sout;
    using core::componentmodel::behavior::LMConstraint<DataTypes,DataTypes>::serr;
    using core::componentmodel::behavior::LMConstraint<DataTypes,DataTypes>::sendl;
protected:
    DistanceConstraintInternalData<DataTypes> data;
    friend class DistanceConstraintInternalData<DataTypes>;

public:
    DistanceConstraint( MechanicalState *dof):
        core::componentmodel::behavior::LMConstraint<DataTypes,DataTypes>(dof,dof),
        vecConstraint(Base::initData(&vecConstraint, "vecConstraint", "List of the edges to constrain"))
    {};
    DistanceConstraint( MechanicalState *dof1, MechanicalState * dof2):
        core::componentmodel::behavior::LMConstraint<DataTypes,DataTypes>(dof1,dof2),
        vecConstraint(Base::initData(&vecConstraint, "vecConstraint", "List of the edges to constrain"))
    {};
    DistanceConstraint():
        vecConstraint(Base::initData(&vecConstraint, "vecConstraint", "List of the edges to constrain")) {}

    ~DistanceConstraint() {};

    // -- Constraint interface
    void init();
    void reinit();
    void writeConstraintEquations(ConstOrder order);

    double getError();

    void addConstraint(unsigned int i1, unsigned int i2);

    virtual void draw();
    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }
    static std::string templateName(const DistanceConstraint<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Edges involving a distance constraint
    Data< SeqEdges > vecConstraint;

protected :
    ///Compute the length of an edge given the vector of coordinates corresponding
    double lengthEdge(const Edge &e, const VecCoord &x1,const VecCoord &x2) const;
    ///Compute the direction of the constraint
    Deriv getDirection(const Edge &e, const VecCoord &x1, const VecCoord &x2) const;
    void updateRestLength();

    // Base Components of the current context
    core::componentmodel::topology::BaseMeshTopology *topology;

    // rest length pre-computated
    sofa::helper::vector< double > l0;
};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
