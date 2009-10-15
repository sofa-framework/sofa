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


template <class DataTypes>
typename DataTypes::Deriv FixedLMConstraint<DataTypes>::getXDirection(unsigned int) {return X;}
template <class DataTypes>
typename DataTypes::Deriv FixedLMConstraint<DataTypes>::getYDirection(unsigned int) {return Y;}
template <class DataTypes>
typename DataTypes::Deriv FixedLMConstraint<DataTypes>::getZDirection(unsigned int) {return Z;}



//Cancel the acceleration (0 along X direction, 0 along Y direction, 0 along Z direction)
template <class DataTypes>
Vector3 FixedLMConstraint<DataTypes>::getExpectedAcceleration(unsigned int ) {return Vector3();}
//Cancel the velocity (0 along X direction, 0 along Y direction, 0 along Z direction)
template <class DataTypes>
Vector3 FixedLMConstraint<DataTypes>::getExpectedVelocity(unsigned int ) {return Vector3();}
//Force the position to the rest position
template <class DataTypes>
Vector3 FixedLMConstraint<DataTypes>::getExpectedPosition(unsigned int index)
{
    Vector3 result;
    //If a new particle has to be fixed, we add its current position as rest position
    if (this->restPosition.find(index) == this->restPosition.end())
    {
        const VecCoord& x = *this->constrainedObject1->getX();
        this->restPosition.insert(std::make_pair(index, x[index]));
    }
    result[0] = this->restPosition[index][0];
    result[1] = this->restPosition[index][1];
    result[2] = this->restPosition[index][2];
    return result;
}





//At init, we store the rest position of the particles we have to fix
template <class DataTypes>
void FixedLMConstraint<DataTypes>::init()
{
    BaseProjectiveLMConstraint<DataTypes>::init();
    initFixedPosition();
    X[0]=1; X[1]=0; X[2]=0;
    Y[0]=0; Y[1]=1; Y[2]=0;
    Z[0]=0; Z[1]=0; Z[2]=1;
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
    //serr<<"FixedLMConstraint<DataTypes>::draw(), x.size() = "<<x.size()<<sendl;


    const SetIndexArray & indices = this->f_indices.getValue().getArray();

    std::vector< Vector3 > points;
    Vector3 point;
    unsigned int sizePoints= (Coord::static_size <=3)?Coord::static_size:3;
    //serr<<"FixedLMConstraint<DataTypes>::draw(), indices = "<<indices<<sendl;
    for (SetIndexArray::const_iterator it = indices.begin();
            it != indices.end();
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


