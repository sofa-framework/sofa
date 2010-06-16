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
#include <sofa/component/projectiveconstraintset/HermiteSplineConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>


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
void HermiteSplineConstraint<Rigid3dTypes>::projectPosition(VecCoord& x)
{
    Real t = (Real) getContext()->getTime();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real DT = m_tEnd.getValue() - m_tBegin.getValue();
        const SetIndexArray & indices = m_indices.getValue().getArray();

        t -= m_tBegin.getValue();
        Real u = t/DT;

        Real H00, H10, H01, H11;
        computeHermiteCoefs( u, H00, H10, H01, H11);

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it].getCenter() = m_x0.getValue()*H00 + m_dx0.getValue()*H10 + m_x1.getValue()*H01 + m_dx1.getValue()*H11;
        }
    }
}

template <>
void HermiteSplineConstraint<Rigid3dTypes>::projectVelocity(VecDeriv& dx)
{
    Real t = (Real) getContext()->getTime();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real DT = m_tEnd.getValue() - m_tBegin.getValue();
        const SetIndexArray & indices = m_indices.getValue().getArray();

        t -= m_tBegin.getValue();
        Real u = t/DT;

        Real dH00, dH10, dH01, dH11;
        computeDerivateHermiteCoefs( u, dH00, dH10, dH01, dH11);

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            dx[*it].getVCenter() = m_x0.getValue()*dH00 + m_dx0.getValue()*dH10 + m_x1.getValue()*dH01 + m_dx1.getValue()*dH11;
        }
    }
}
#endif
#ifndef SOFA_DOUBLE
template <>
void HermiteSplineConstraint<Rigid3fTypes>::projectPosition(VecCoord& x)
{
    Real t = (Real) getContext()->getTime();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real DT = m_tEnd.getValue() - m_tBegin.getValue();
        const SetIndexArray & indices = m_indices.getValue().getArray();

        t -= m_tBegin.getValue();
        Real u = t/DT;

        Real H00, H10, H01, H11;
        computeHermiteCoefs( u, H00, H10, H01, H11);

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it].getCenter() = m_x0.getValue()*H00 + m_dx0.getValue()*H10 + m_x1.getValue()*H01 + m_dx1.getValue()*H11;
        }
    }
}
template <>
void HermiteSplineConstraint<Rigid3fTypes>::projectVelocity(VecDeriv& dx)
{
    Real t = (Real) getContext()->getTime();

    if ( t >= m_tBegin.getValue() && t <= m_tEnd.getValue()	)
    {
        Real DT = m_tEnd.getValue() - m_tBegin.getValue();
        const SetIndexArray & indices = m_indices.getValue().getArray();

        t -= m_tBegin.getValue();
        Real u = t/DT;

        Real dH00, dH10, dH01, dH11;
        computeDerivateHermiteCoefs( u, dH00, dH10, dH01, dH11);

        for(SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            dx[*it].getVCenter() = m_x0.getValue()*dH00 + m_dx0.getValue()*dH10 + m_x1.getValue()*dH01 + m_dx1.getValue()*dH11;
        }
    }
}
#endif






SOFA_DECL_CLASS(HermiteSplineConstraint)


int HermiteSplineConstraintClass = core::RegisterObject("Apply a hermite cubic spline trajectory to given points")
#ifndef SOFA_FLOAT
        .add< HermiteSplineConstraint<Vec3dTypes> >()
        .add< HermiteSplineConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< HermiteSplineConstraint<Vec3fTypes> >()
        .add< HermiteSplineConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class HermiteSplineConstraint<Rigid3dTypes>;
template class HermiteSplineConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class HermiteSplineConstraint<Rigid3fTypes>;
template class HermiteSplineConstraint<Vec3fTypes>;
#endif



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

