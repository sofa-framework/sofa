/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/constraint/OscillatorConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
//#include <sofa/helper/gl/Axis.h>
#include <sstream>

namespace sofa
{

namespace component
{

namespace constraint
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

template class OscillatorConstraint<Vec3dTypes>;
template class OscillatorConstraint<Vec3fTypes>;
template class OscillatorConstraint<Rigid3dTypes>;
template class OscillatorConstraint<Rigid3fTypes>;

int OscillatorConstraintClass = core::RegisterObject("Apply a sinusoidal trajectory to given points")
        .add< OscillatorConstraint<Vec3dTypes> >()
        .add< OscillatorConstraint<Vec3fTypes> >()
        .add< OscillatorConstraint<Rigid3dTypes> >()
        .add< OscillatorConstraint<Rigid3fTypes> >()
        ;

} // namespace constraint

} // namespace component

} // namespace sofa

