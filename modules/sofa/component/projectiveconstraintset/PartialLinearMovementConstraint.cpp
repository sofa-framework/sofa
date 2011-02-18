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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARTIALLINEARMOVEMENTCONSTRAINT_CPP
#include <sofa/component/projectiveconstraintset/PartialLinearMovementConstraint.inl>
#include <sofa/core/behavior/Constraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


//display specialisation for rigid types
#ifndef SOFA_FLOAT
template <>
void PartialLinearMovementConstraint<Rigid3dTypes>::draw()
{
    const SetIndexArray & indices = m_indices.getValue().getArray();
    if (!getContext()->getShowBehaviorModels()) return;
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_LINES);
    for (unsigned int i=0 ; i<m_keyMovements.getValue().size()-1 ; i++)
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            gl::glVertexT(x0[*it].getCenter()+m_keyMovements.getValue()[i].getVCenter());
            gl::glVertexT(x0[*it].getCenter()+m_keyMovements.getValue()[i+1].getVCenter());
        }
    }
    glEnd();
}

template <>
void PartialLinearMovementConstraint<Rigid3dTypes>::projectPosition(VecCoord& x)
{
    Real cT = (Real) this->getContext()->getTime();

    //initialize initial Dofs positions, if it's not done
    if (x0.size() == 0)
    {
        const SetIndexArray & indices = m_indices.getValue().getArray();
        x0.resize( x.size() );
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            x0[*it] = x[*it];
    }

    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    //if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
    if(finished && nextT != prevT)
    {
        const SetIndexArray & indices = m_indices.getValue().getArray();

        Real dt = (cT - prevT) / (nextT - prevT);
        Deriv m = prevM + (nextM-prevM)*dt;
        Quater<double> prevOrientation = Quater<double>::createQuaterFromEuler(prevM.getVOrientation());
        Quater<double> nextOrientation = Quater<double>::createQuaterFromEuler(nextM.getVOrientation());
        //set the motion to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it].getCenter() = x0[*it].getCenter() + m.getVCenter() ;
            x[*it].getOrientation() = x0[*it].getOrientation() * prevOrientation.slerp2(nextOrientation, dt);
        }
    }
}

#endif


#ifndef SOFA_DOUBLE
template <>
void PartialLinearMovementConstraint<Rigid3fTypes>::draw()
{
    const SetIndexArray & indices = m_indices.getValue().getArray();
    if (!getContext()->getShowBehaviorModels()) return;
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_LINES);
    for (unsigned int i=0 ; i<m_keyMovements.getValue().size()-1 ; i++)
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            gl::glVertexT(x0[*it].getCenter()+m_keyMovements.getValue()[i].getVCenter());
            gl::glVertexT(x0[*it].getCenter()+m_keyMovements.getValue()[i+1].getVCenter());
        }
    }
    glEnd();
}

template <>
void PartialLinearMovementConstraint<Rigid3fTypes>::projectPosition(VecCoord& x)
{
    Real cT = (Real) this->getContext()->getTime();

    //initialize initial Dofs positions, if it's not done
    if (x0.size() == 0)
    {
        const SetIndexArray & indices = m_indices.getValue().getArray();
        x0.resize( x.size() );
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            x0[*it] = x[*it];
    }

    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    //if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
    if(finished && nextT != prevT)
    {
        const SetIndexArray & indices = m_indices.getValue().getArray();

        Real dt = (cT - prevT) / (nextT - prevT);
        Deriv m = prevM + (nextM-prevM)*dt;
        Quater<double> prevOrientation = Quater<double>::createQuaterFromEuler(prevM.getVOrientation());
        Quater<double> nextOrientation = Quater<double>::createQuaterFromEuler(nextM.getVOrientation());

        //set the motion to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it].getCenter() = x0[*it].getCenter() + m.getVCenter() ;
            x[*it].getOrientation() = x0[*it].getOrientation() * prevOrientation.slerp2(nextOrientation, dt);
        }
    }
}

#endif

//declaration of the class, for the factory
SOFA_DECL_CLASS(PartialLinearMovementConstraint)


int PartialLinearMovementConstraintClass = core::RegisterObject("translate given particles")
#ifndef SOFA_FLOAT
        .add< PartialLinearMovementConstraint<Vec3dTypes> >()
        .add< PartialLinearMovementConstraint<Vec2dTypes> >()
        .add< PartialLinearMovementConstraint<Vec1dTypes> >()
        .add< PartialLinearMovementConstraint<Vec6dTypes> >()
        .add< PartialLinearMovementConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PartialLinearMovementConstraint<Vec3fTypes> >()
        .add< PartialLinearMovementConstraint<Vec2fTypes> >()
        .add< PartialLinearMovementConstraint<Vec1fTypes> >()
        .add< PartialLinearMovementConstraint<Vec6fTypes> >()
        .add< PartialLinearMovementConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Vec3dTypes>;
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Vec2dTypes>;
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Vec1dTypes>;
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Vec6dTypes>;// Phuoc
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Vec3fTypes>;
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Vec2fTypes>;
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Vec1fTypes>;
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Vec6fTypes>; //Phuoc
template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API PartialLinearMovementConstraint<Rigid3fTypes>;
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa
