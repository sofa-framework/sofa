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
#ifndef SOFA_COMPONENT_CONSTRAINT_FIXEDLMCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_FIXEDLMCONSTRAINT_INL

#include <sofa/component/constraint/FixedLMConstraint.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>





namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::helper;


//Cancel the acceleration (0 along X direction, 0 along Y direction, 0 along Z direction)
template <class DataTypes>
void FixedLMConstraint<DataTypes>::getExpectedAcceleration(unsigned int, helper::vector< SReal >&expectedValue  )
{
    expectedValue[0]=expectedValue[1]=expectedValue[2]=0;
}
//Cancel the velocity (0 along X direction, 0 along Y direction, 0 along Z direction)
template <class DataTypes>
void FixedLMConstraint<DataTypes>::getExpectedVelocity(unsigned int, helper::vector< SReal >&expectedValue )
{
    expectedValue[0]=expectedValue[1]=expectedValue[2]=0;
}

//Force the position to the rest position
template <class DataTypes>
void FixedLMConstraint<DataTypes>::getPositionCorrection(unsigned int index, helper::vector< SReal > &correction)
{
    const VecCoord& x = *this->constrainedObject1->getX();
    //If a new particle has to be fixed, we add its current position as rest position
    if (this->restPosition.find(index) == this->restPosition.end())
    {
        this->restPosition.insert(std::make_pair(index, x[index]));
    }

    // We want to correct the position so that the current position be equal to the rest position
    correction[0] = this->restPosition[index][0]-x[index][0];
    correction[1] = this->restPosition[index][1]-x[index][1];
    correction[2] = this->restPosition[index][2]-x[index][2];
}





//At init, we store the rest position of the particles we have to fix
template <class DataTypes>
void FixedLMConstraint<DataTypes>::init()
{
    BaseProjectiveLMConstraint<DataTypes>::init();
    initFixedPosition();

    //the constraint can be applied for the three orders
    this->usingACC=this->usingVEL=this->usingPOS=true;

    //Define the direction of the constraints
    //We will constrain the 3 degrees of freedom of translation X,Y,Z
    this->constraintDirection.resize(3);

    Deriv X,Y,Z;
    X[0]=1; X[1]=0; X[2]=0;
    Y[0]=0; Y[1]=1; Y[2]=0;
    Z[0]=0; Z[1]=0; Z[2]=1;

    this->constraintDirection[0]=X;
    this->constraintDirection[1]=Y;
    this->constraintDirection[2]=Z;
}


template <class DataTypes>
void FixedLMConstraint<DataTypes>::initFixedPosition()
{
    this->restPosition.clear();
    const VecCoord& x = *this->constrainedObject1->getX();
    const SetIndexArray & indices = this->f_indices.getValue().getArray();
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        unsigned int index=*it;
        this->restPosition.insert(std::make_pair(index, x[index]));
    }
}


template <class DataTypes>
void FixedLMConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels()) return;
    const VecCoord& x = *this->constrainedObject1->getX();

    const SetIndexArray & indices = this->f_indices.getValue().getArray();

    std::vector< Vector3 > points;
    Vector3 point;
    unsigned int sizePoints= (Coord::static_size <=3)?Coord::static_size:3;

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end();
            ++it)
    {
        for (unsigned int s=0; s<sizePoints; ++s) point[s] = x[*it][s];
        points.push_back(point);
    }
    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        simulation::getSimulation()->DrawUtility.drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    }
    else
    {
        simulation::getSimulation()->DrawUtility.drawSpheres(points, (float)_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
    }
}

// Specialization for rigids
#ifndef SOFA_FLOAT
template <>
void FixedLMConstraint<Rigid3dTypes >::draw();
#endif
#ifndef SOFA_DOUBLE
template <>
void FixedLMConstraint<Rigid3fTypes >::draw();
#endif





} // namespace constraint

} // namespace component

} // namespace sofa

#endif


