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
#ifndef SOFA_COMPONENT_CONSTRAINT_FIXEDLMCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_FIXEDLMCONSTRAINT_H

#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/component/constraint/BaseProjectiveLMConstraint.h>


namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::core::componentmodel::topology;
/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class FixedLMConstraintInternalData
{
};




/** Keep a set of particle at a fixed position
 */
template <class DataTypes>
class FixedLMConstraint :  public BaseProjectiveLMConstraint<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;


    typedef sofa::component::topology::PointSubset SetIndex;
    typedef helper::vector<unsigned int> SetIndexArray;
protected:
    FixedLMConstraintInternalData<DataTypes> data;
    friend class FixedLMConstraintInternalData<DataTypes>;

public:
    FixedLMConstraint( MechanicalState *dof): BaseProjectiveLMConstraint<DataTypes>(dof),
        _drawSize(core::objectmodel::Base::initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    {};
    FixedLMConstraint():
        _drawSize(core::objectmodel::Base::initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    {}

    Deriv getExpectedAcceleration(unsigned int index);
    Deriv getExpectedVelocity    (unsigned int index);
    Coord getExpectedPosition    (unsigned int index);

    void init();
    void initFixedPosition();
    void reset() {initFixedPosition();};
    void draw();

protected :

    std::map< unsigned int, Coord> restPosition;
    Data<double> _drawSize;

};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
