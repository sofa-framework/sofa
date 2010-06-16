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
#include <sofa/component/projectiveconstraintset/ParabolicConstraint.inl>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
//#include <sofa/helper/gl/Axis.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


#ifndef SOFA_FLOAT
template <>
void ParabolicConstraint<Rigid3dTypes>::projectPosition(VecCoord& x)
{
    Real t = (Real) getContext()->getTime();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real relativeTime = (t - m_tBegin.getValue() ) / (m_tEnd.getValue() - m_tBegin.getValue());
        const SetIndexArray & indices = m_indices.getValue().getArray();

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            Real px = m_locP3.x()*relativeTime;
            Real py = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (px *px) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * px;
            Vec3R locPos( px , py, 0.0);

            Vec3R worldPos = m_P1.getValue() + m_projection.rotate(locPos);

            x[*it].getCenter() = worldPos;
        }
    }
}

template <>
void ParabolicConstraint<Rigid3dTypes>::projectVelocity(VecDeriv& dx)
{
    Real t = (Real) getContext()->getTime();
    Real dt = (Real) getContext()->getDt();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real relativeTime = (t - m_tBegin.getValue() ) / (m_tEnd.getValue() - m_tBegin.getValue());
        const SetIndexArray & indices = m_indices.getValue().getArray();

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            Real pxP = m_locP3.x()*relativeTime;
            Real pyP = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (pxP *pxP) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * pxP;
            relativeTime = (t+dt - m_tBegin.getValue() ) / (m_tEnd.getValue() - m_tBegin.getValue());
            Real pxN = m_locP3.x()*relativeTime;
            Real pyN = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (pxN *pxN) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * pxN;

            Vec3R locVel = Vec3R( (pxN-pxP)/dt, (pyN-pyP)/dt, 0.0);

            Vec3R worldVel = m_projection.rotate(locVel);

            dx[*it].getVCenter() = worldVel;
        }
    }
}
#endif
#ifndef SOFA_DOUBLE
template <>
void ParabolicConstraint<Rigid3fTypes>::projectPosition(VecCoord& x)
{
    Real t = (Real) getContext()->getTime();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real relativeTime = (t - m_tBegin.getValue() ) / (m_tEnd.getValue() - m_tBegin.getValue());
        const SetIndexArray & indices = m_indices.getValue().getArray();

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            Real px = m_locP3.x()*relativeTime;
            Real py = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (px *px) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * px;
            Vec3R locPos( px , py, 0.0);

            Vec3R worldPos = m_P1.getValue() + m_projection.rotate(locPos);

            x[*it].getCenter() = worldPos;
        }
    }
}
template <>
void ParabolicConstraint<Rigid3fTypes>::projectVelocity(VecDeriv& dx)
{
    Real t = (Real) getContext()->getTime();
    Real dt = (Real) getContext()->getDt();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real relativeTime = (t - m_tBegin.getValue() ) / (m_tEnd.getValue() - m_tBegin.getValue());
        const SetIndexArray & indices = m_indices.getValue().getArray();

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            Real pxP = m_locP3.x()*relativeTime;
            Real pyP = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (pxP *pxP) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * pxP;
            relativeTime = (t+dt - m_tBegin.getValue() ) / (m_tEnd.getValue() - m_tBegin.getValue());
            Real pxN = m_locP3.x()*relativeTime;
            Real pyN = (- m_locP2.y() / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * (pxN *pxN) + ( (m_locP3.x()*m_locP2.y()) / (m_locP3.x()*m_locP2.x() - m_locP2.x()*m_locP2.x())) * pxN;

            Vec3R locVel = Vec3R( (pxN-pxP)/dt, (pyN-pyP)/dt, 0.0);

            Vec3R worldVel = m_projection.rotate(locVel);

            dx[*it].getVCenter() = worldVel;
        }
    }
}
#endif






SOFA_DECL_CLASS(ParabolicConstraint)


int ParabolicConstraintClass = core::RegisterObject("Apply a parabolic trajectory to given points")
#ifndef SOFA_FLOAT
        .add< ParabolicConstraint<Vec3dTypes> >()
        .add< ParabolicConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ParabolicConstraint<Vec3fTypes> >()
        .add< ParabolicConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class ParabolicConstraint<Rigid3dTypes>;
template class ParabolicConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class ParabolicConstraint<Rigid3fTypes>;
template class ParabolicConstraint<Vec3fTypes>;
#endif



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

