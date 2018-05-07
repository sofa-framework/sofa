/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "RayTriangleVisitor.h"
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaOpenglVisual/OglModel.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

using core::objectmodel::BaseObject;

namespace component{
namespace collision {


RayTriangleVisitor::RayTriangleVisitor(const core::ExecParams* params):simulation::Visitor(params)
{}

struct distanceHitSort{
    bool operator() (const RayTriangleVisitor::Hit& h1, const RayTriangleVisitor::Hit& h2){
        return h1.distance < h2.distance;
    }
};

core::objectmodel::BaseObject* RayTriangleVisitor::embeddingModel()
{
    std::sort(hits.begin(),hits.end(),distanceHitSort());
    //================================================================
    // Version 2: we assume that objects are not nested, there can be intersections
    // We look for the first exit from an object not previously entered in
    std::set<BaseObject*> entered;
    for( std::size_t i= 0; i<hits.size(); i++ )
    {
        if( !hits[i].internal ) {
            // entering an object
            entered.insert(hits[i].hitObject);
        }
        else {
            // leaving an object
            if( entered.find(hits[i].hitObject )==entered.end() ){
                // not previously entered
                return hits[i].hitObject;
            }
        }
    }
    return NULL;
}

void RayTriangleVisitor::processTriangleModel(simulation::Node* /*node*/, component::collision::TriangleModel* tm)
{
    typedef component::collision::TriangleModel TriangleModel;
    typedef TriangleModel::DataTypes DataTypes;

    const DataTypes::VecCoord& x = tm->getMechanicalState()->read(sofa::core::ConstVecCoordId::position())->getValue();
    for( core::topology::BaseMeshTopology::SeqTriangles::const_iterator it=tm->getTriangles().begin(), iend=tm->getTriangles().end() ; it!=iend; it++)
    {
        const Vec3& v0 = x[(*it)[0]];
        const Vec3& v1 = x[(*it)[1]];
        const Vec3& v2 = x[(*it)[2]];

        // ray-triangle intersection adapted from http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-9-ray-triangle-intersection/ray-triangle-intersection-geometric-solution/
        Vec3 e01 = v1-v0;
        Vec3 e02 = v2-v0;
        Vec3 N = e01.cross(e02);

        // find intersection point
        SReal NdotRayDir = N*direction;
        SReal eps = std::numeric_limits<SReal>::epsilon() * 100;
        if( fabs(NdotRayDir)<eps) continue;  // ray parallel to plane.
        SReal d= N*v0;
        SReal t= -(N*origin + d)/NdotRayDir;
        if( t<0 ) continue; // the triangle is behind
        Vec3 P = origin + t * direction;

        // inside-outside test, edge 01
        Vec3 VP0 = P-v0;
        Vec3 C = e01.cross(VP0);
        if( N*C<0 ) continue; // point on the right side of the edge

        // inside-outside test, edge 12
        Vec3 VP1 = P-v1;
        Vec3 e12 = v2-v1;
        C = e12.cross(VP1);
        if( N*C<0 ) continue; // point on the right side of the edge

        // inside-outside test, edge 20
        Vec3 VP2 = P-v2;
        C = e02.cross(VP2);
        if( N*C>0 ) continue; // point on the left side of the edge

        Hit hit;
        hit.hitObject = tm;
        hit.distance = (P-origin).norm();
        hit.internal = (N*direction)<0;
        hits.push_back( hit );
    }
}

void RayTriangleVisitor::processOglModel(simulation::Node* /*node*/, component::visualmodel::OglModel* om)
{
    using component::visualmodel::OglModel;
    typedef defaulttype::ExtVec3fTypes DataTypes;

    const DataTypes::VecCoord& x = om->getVertices();
    for( std::size_t i=0; i<om->getTriangles().size(); i++ )
    {
        const Vec3& v0 = x[om->getTriangles()[i][0]];
        const Vec3& v1 = x[om->getTriangles()[i][1]];
        const Vec3& v2 = x[om->getTriangles()[i][2]];

        // ray-triangle intersection adapted from http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-9-ray-triangle-intersection/ray-triangle-intersection-geometric-solution/
        Vec3 e01 = v1-v0;
        Vec3 e02 = v2-v0;
        Vec3 N = e01.cross(e02);

        // find intersection point
        SReal NdotRayDir = N*direction;
        SReal eps = std::numeric_limits<SReal>::epsilon() * 100;
        if( fabs(NdotRayDir)<eps) continue;  // ray parallel to plane.
        SReal d= N*v0;
        SReal t= -(N*origin + d)/NdotRayDir;
        if( t<0 ) continue; // the triangle is behind
        Vec3 P = origin + t * direction;

        // inside-outside test, edge 01
        Vec3 VP0 = P-v0;
        Vec3 C = e01.cross(VP0);
        if( N*C<0 ) continue; // point on the right side of the edge

        // inside-outside test, edge 12
        Vec3 VP1 = P-v1;
        Vec3 e12 = v2-v1;
        C = e12.cross(VP1);
        if( N*C<0 ) continue; // point on the right side of the edge

        // inside-outside test, edge 20
        Vec3 VP2 = P-v2;
        C = e02.cross(VP2);
        if( N*C>0 ) continue; // point on the left side of the edge

        Hit hit;
        hit.hitObject = om;
        hit.distance = (P-origin).norm();
        hit.internal = (N*direction)<0;
        hits.push_back( hit );

    }
}

simulation::Visitor::Result RayTriangleVisitor::processNodeTopDown(simulation::Node* node)
{
    using core::visual::VisualModel;
    using sofa::component::visualmodel::OglModel;
    typedef simulation::Node::Sequence<core::CollisionModel> CollisionModels;
    typedef simulation::Node::Sequence<VisualModel> VisualModels;
    using component::collision::TriangleModel;

    for( CollisionModels::const_iterator it=node->collisionModel.begin(), iend=node->collisionModel.end(); it!=iend; it++ )
    {
        if( TriangleModel* tmodel = dynamic_cast<TriangleModel*>(*it) ) {
            processTriangleModel(node,tmodel);
        }
    }
    for( VisualModels::const_iterator it=node->visualModel.begin(), iend=node->visualModel.end(); it!=iend; it++ )
    {
        if( OglModel* tmodel = dynamic_cast<OglModel*>(*it) ) {
            processOglModel(node,tmodel);
        }
    }

    return Visitor::RESULT_CONTINUE;
}


}
}
} // namespace sofa

