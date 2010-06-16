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
#include <sofa/component/projectiveconstraintset/OscillatorConstraint.inl>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
//#include <sofa/helper/gl/Axis.h>
#include <sstream>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

//template <>
//void OscillatorConstraint<RigidTypes>::draw()
//{
///*	if (!getContext()->getShowBehaviorModels()) return;
//	VecCoord& x = *mstate->getX();
//	for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
//	{
//		int i = *it;
//		Quat orient = x[i].getOrientation();
//		RigidTypes::Vec3& center = x[i].getCenter();
//		orient[3] = -orient[3];
//
//		static GL::Axis *axis = new GL::Axis(center, orient, 5);
//
//		axis->update(center, orient);
//		axis->draw();
//	}*/
//	if (!getContext()->getShowBehaviorModels()) return;
//	VecCoord& x = *mstate->getX();
//	glDisable (GL_LIGHTING);
//	glPointSize(10);
//	glColor4f (1,0.5,0.5,1);
//	glBegin (GL_POINTS);
//	for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
//	{
//	   helper::gl::glVertexT(x[0].getCenter());
//	}
//	glEnd();
//}
//
//template <>
//      void OscillatorConstraint<RigidTypes>::projectResponse(VecDeriv& res)
//{
//   res[0] = Deriv();
//}


SOFA_DECL_CLASS(OscillatorConstraint)


int OscillatorConstraintClass = core::RegisterObject("Apply a sinusoidal trajectory to given points")
#ifndef SOFA_FLOAT
        .add< OscillatorConstraint<Vec3dTypes> >()
        .add< OscillatorConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< OscillatorConstraint<Vec3fTypes> >()
        .add< OscillatorConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class OscillatorConstraint<Rigid3dTypes>;
template class OscillatorConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class OscillatorConstraint<Rigid3fTypes>;
template class OscillatorConstraint<Vec3fTypes>;
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

