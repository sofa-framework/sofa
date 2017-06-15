/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPlaneConstraint_CPP
#include <SofaBoundaryCondition/ProjectToPlaneConstraint.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace sofa::defaulttype;
using namespace sofa::helper;


SOFA_DECL_CLASS(ProjectToPlaneConstraint)

int ProjectToPlaneConstraintClass = core::RegisterObject("Attach given particles to their initial positions")
#ifndef SOFA_FLOAT
        .add< ProjectToPlaneConstraint<Vec3dTypes> >()
        .add< ProjectToPlaneConstraint<Vec2dTypes> >()
//.add< ProjectToPlaneConstraint<Vec1dTypes> >()
//.add< ProjectToPlaneConstraint<Vec6dTypes> >()
//.add< ProjectToPlaneConstraint<Rigid3dTypes> >()
//.add< ProjectToPlaneConstraint<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ProjectToPlaneConstraint<Vec3fTypes> >()
        .add< ProjectToPlaneConstraint<Vec2fTypes> >()
//.add< ProjectToPlaneConstraint<Vec1fTypes> >()
//.add< ProjectToPlaneConstraint<Vec6fTypes> >()
//.add< ProjectToPlaneConstraint<Rigid3fTypes> >()
//.add< ProjectToPlaneConstraint<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Vec2dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Vec1dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Vec6dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Rigid3dTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Vec2fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Vec1fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Vec6fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Rigid3fTypes>;
//template class SOFA_BOUNDARY_CONDITION_API ProjectToPlaneConstraint<Rigid2fTypes>;
#endif

//#ifndef SOFA_FLOAT
//template <>
//void ProjectToPlaneConstraint<Rigid3dTypes>::draw(const core::visual::VisualParams* vparams)
//{
//        const SetIndexArray & indices = f_indices.getValue();
//	if (!vparams->displayFlags().getShowBehaviorModels()) return;
//	std::vector< Vector3 > points;

//	const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
//	if( f_fixAll.getValue()==true )
//	    for (unsigned i=0; i<x.size(); i++ )
//              points.push_back(x[i].getCenter());
//	else
//	{
//		if( x.size() < indices.size() )
//		{
//			for (unsigned i=0; i<x.size(); i++ )
//              points.push_back(x[indices[i]].getCenter());
//		}
//		else
//		{
//			for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
//				  points.push_back(x[*it].getCenter());
//		}
//	}

//        if( _drawSize.getValue() == 0) // old classical drawing by points
//          vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
//        else
//          vparams->drawTool()->drawSpheres(points, (float)_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
//}

//template <>
//    void ProjectToPlaneConstraint<Rigid2dTypes>::draw(const core::visual::VisualParams* vparams)
//{
//  const SetIndexArray & indices = f_indices.getValue();
//  if (!vparams->displayFlags().getShowBehaviorModels()) return;
//  std::vector< Vector3 > points;

//  const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
//  glDisable (GL_LIGHTING);
//  glPointSize(10);
//  glColor4f (1,0.5,0.5,1);
//  glBegin (GL_POINTS);
//  if( f_fixAll.getValue()==true )
//    for (unsigned i=0; i<x.size(); i++ )
//      gl::glVertexT(x[i].getCenter());
//  else
//    for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
//      gl::glVertexT(x[*it].getCenter());
//  glEnd();
//  glPointSize(1);
//}
//#endif

//#ifndef SOFA_DOUBLE
//template <>
//void ProjectToPlaneConstraint<Rigid3fTypes>::draw(const core::visual::VisualParams* vparams)
//{
//        const SetIndexArray & indices = f_indices.getValue();
//	if (!vparams->displayFlags().getShowBehaviorModels()) return;
//	std::vector< Vector3 > points;

//	const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
//	if( f_fixAll.getValue()==true )
//	    for (unsigned i=0; i<x.size(); i++ )
//              points.push_back(x[i].getCenter());
//	else
//	    for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
//              points.push_back(x[*it].getCenter());

//        if( _drawSize.getValue() == 0) // old classical drawing by points
//          vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
//        else
//          vparams->drawTool()->drawSpheres(points, (float)_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
//}

//template <>
//    void ProjectToPlaneConstraint<Rigid2fTypes>::draw(const core::visual::VisualParams* vparams)
//{
//  const SetIndexArray & indices = f_indices.getValue();
//  if (!vparams->displayFlags().getShowBehaviorModels()) return;
//  const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
//  glDisable (GL_LIGHTING);
//  glPointSize(10);
//  glColor4f (1,0.5,0.5,1);
//  glBegin (GL_POINTS);
//  if( f_fixAll.getValue()==true )
//    for (unsigned i=0; i<x.size(); i++ )
//      gl::glVertexT(x[i].getCenter());
//  else
//    for (SetIndex::const_iterator it = indices.begin(); it != indices.end(); ++it)
//      gl::glVertexT(x[*it].getCenter());
//  glEnd();
//  glPointSize(1);
//}
//#endif



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

