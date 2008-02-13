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
#include <sofa/component/collision/BarycentricStickContact.inl>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(BarycentricStickContact)

Creator<Contact::Factory, BarycentricStickContact<SphereModel, SphereModel> > SphereSphereStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<SphereModel, PointModel> > SpherePointStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<SphereTreeModel, SphereTreeModel> > SphereTreeSphereTreeStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<SphereTreeModel, TriangleMeshModel> > SphereTreeTriangleMeshStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<SphereTreeModel, TriangleSetModel> > SphereTreeTriangleSetStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<PointModel, PointModel> > PointPointStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<LineMeshModel, PointModel> > LinePointStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<LineMeshModel, LineMeshModel> > LineLineStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<LineMeshModel, SphereModel> > LineSphereStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleMeshModel, SphereModel> > TriangleMeshSphereStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleMeshModel, PointModel> > TriangleMeshPointStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleMeshModel, LineMeshModel> > TriangleMeshLineStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleMeshModel, TriangleMeshModel> > TriangleMeshTriangleMeshStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleMeshModel, TriangleSetModel> > TriangleMeshTriangleSetStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleSetModel, SphereModel> > TriangleSetSphereStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleSetModel, PointModel> > TriangleSetPointStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleSetModel, LineMeshModel> > TriangleSetLineStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleSetModel, TriangleMeshModel> > TriangleSetTriangleMeshStickContactClass("stick",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleSetModel, TriangleSetModel> > TriangleSetTriangleSetStickContactClass("stick",true);

Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, PointModel> > DistanceGridPointStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, SphereModel> > DistanceGridSphereStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, TriangleMeshModel> > DistanceGridTriangleMeshStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, TriangleSetModel> > DistanceGridTriangleSetStickContactClass("stick", true);

Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, PointModel> > FFDDistanceGridPointStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, SphereModel> > FFDDistanceGridSphereStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, TriangleMeshModel> > FFDDistanceGridTriangleMeshStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, TriangleSetModel> > FFDDistanceGridTriangleSetStickContactClass("stick", true);


} // namespace collision

} // namespace component

} // namespace sofa

