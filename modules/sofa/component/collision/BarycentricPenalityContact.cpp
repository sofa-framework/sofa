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
#include <sofa/component/collision/BarycentricPenalityContact.inl>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using simulation::tree::GNode;

SOFA_DECL_CLASS(BarycentricPenalityContact)

Creator<Contact::Factory, BarycentricPenalityContact<SphereModel, SphereModel> > SphereSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereModel, PointModel> > SpherePointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereTreeModel, SphereTreeModel> > SphereTreeSphereTreeContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereTreeModel, TriangleMeshModel> > SphereTreeTriangleMeshContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereTreeModel, TriangleSetModel> > SphereTreeTriangleSetContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<PointModel, PointModel> > PointPointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineModel, PointModel> > LinePointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineModel, LineModel> > LineLinePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineModel, SphereModel> > LineSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleMeshModel, SphereModel> > TriangleMeshSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleMeshModel, PointModel> > TriangleMeshPointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleMeshModel, LineModel> > TriangleMeshLinePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleMeshModel, TriangleMeshModel> > TriangleMeshTriangleMeshPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleMeshModel, TriangleSetModel> > TriangleMeshTriangleSetPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleSetModel, SphereModel> > TriangleSetSpherePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleSetModel, PointModel> > TriangleSetPointPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleSetModel, LineModel> > TriangleSetLinePenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleSetModel, TriangleMeshModel> > TriangleSetTriangleMeshPenalityContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleSetModel, TriangleSetModel> > TriangleSetTriangleSetPenalityContactClass("default",true);

Creator<Contact::Factory, BarycentricPenalityContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidDistanceGridCollisionModel, PointModel> > DistanceGridPointContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidDistanceGridCollisionModel, SphereModel> > DistanceGridSphereContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidDistanceGridCollisionModel, TriangleMeshModel> > DistanceGridTriangleMeshContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidDistanceGridCollisionModel, TriangleSetModel> > DistanceGridTriangleSetContactClass("default", true);

Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, PointModel> > FFDDistanceGridPointContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, SphereModel> > FFDDistanceGridSphereContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, TriangleMeshModel> > FFDDistanceGridTriangleMeshContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, TriangleSetModel> > FFDDistanceGridTriangleSetContactClass("default", true);


} // namespace collision

} // namespace component

} // namespace sofa

