/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/gui/component/performer/InciseAlongPathPerformer.h>

#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/Factory.inl>

namespace sofa::gui::component::performer
{
helper::Creator<InteractionPerformer::InteractionPerformerFactory, InciseAlongPathPerformer>  InciseAlongPathPerformerClass("InciseAlongPath");


void InciseAlongPathPerformer::start()
{
    startBody=this->interactor->getBodyPicked();

    if (startBody.body == 0)
        return;

    if (cpt == 0) // register first position of incision
    {
        firstIncisionBody = startBody;
        cpt++;
        initialNbTriangles = startBody.body->getCollisionTopology()->getNbTriangles();
        initialNbPoints = startBody.body->getCollisionTopology()->getNbPoints();
    }
}

void InciseAlongPathPerformer::execute()
{

    if (freezePerformer) // This performer has been freezed
    {
        if (currentMethod == 1)
            startBody=this->interactor->getBodyPicked();

        return;
    }

    if (currentMethod == 0) // incise from clic to clic
    {
        if (firstBody.body == nullptr) // first clic
            firstBody=startBody;
        else
        {
            if (firstBody.indexCollisionElement != startBody.indexCollisionElement)
                secondBody=startBody;
        }


        if (firstBody.body == nullptr || secondBody.body == nullptr) return;

        sofa::core::topology::TopologyModifier* topologyModifier;
        firstBody.body->getContext()->get(topologyModifier);

        // Handle Removing of topological element (from any type of topology)
        if(topologyModifier)
        {
            topologyChangeManager.incisionCollisionModel(firstBody.body, firstBody.indexCollisionElement, firstBody.point,
                    secondBody.body,  secondBody.indexCollisionElement,  secondBody.point,
                    snapingValue, snapingBorderValue );
        }

        firstBody = secondBody;
        secondBody.body = nullptr;

        this->interactor->setBodyPicked(secondBody);
    }
    else
    {

        BodyPicked currentBody=this->interactor->getBodyPicked();
        if (currentBody.body == nullptr || startBody.body == nullptr) return;

        if (currentBody.indexCollisionElement == startBody.indexCollisionElement) return;

        sofa::core::topology::TopologyModifier* topologyModifier;
        startBody.body->getContext()->get(topologyModifier);

        // Handle Removing of topological element (from any type of topology)
        if(topologyModifier)
        {
            topologyChangeManager.incisionCollisionModel(startBody.body, startBody.indexCollisionElement, startBody.point,
                    currentBody.body,  currentBody.indexCollisionElement,  currentBody.point,
                    snapingValue, snapingBorderValue );
        }
        startBody = currentBody;
        firstBody = currentBody;

        currentBody.body=nullptr;
        this->interactor->setBodyPicked(currentBody);
    }

}

void InciseAlongPathPerformer::setPerformerFreeze()
{
    freezePerformer = true;
    if (fullcut)
        this->PerformCompleteIncision();

    fullcut = true;
}

void InciseAlongPathPerformer::PerformCompleteIncision()
{
    if (firstIncisionBody.body == nullptr || startBody.body == nullptr)
    {
        msg_error("InciseAlongPathPerformer") << "One picked body is null." ;
        return;
    }


    if (firstIncisionBody.indexCollisionElement == startBody.indexCollisionElement)
    {
        msg_error("InciseAlongPathPerformer") << "Picked body are the same." ;
        return;
    }

    // Initial point could have move due to gravity: looking for new coordinates of first incision point and triangle index.
    bool findTri = false;
    const auto& triAroundVertex = startBody.body->getCollisionTopology()->getTrianglesAroundVertex(initialNbPoints);

    // Check if point index and triangle index are consistent.
    for (unsigned int j = 0; j<triAroundVertex.size(); ++j)
        if (triAroundVertex[j] == initialNbTriangles)
        {
            findTri = true;
            break;
        }

    if (!findTri)
    {
        dmsg_error("InciseAlongPathPerformer") << " initial point of incision has not been found." ;
        return;
    }


    // Get new coordinate of first incision point:
    sofa::component::statecontainer::MechanicalObject<defaulttype::Vec3Types>* MechanicalObject=nullptr;
    startBody.body->getContext()->get(MechanicalObject, sofa::core::objectmodel::BaseContext::SearchRoot);
    const auto& positions = MechanicalObject->read(core::ConstVecCoordId::position())->getValue();
    const sofa::type::Vec3& the_point = positions[initialNbPoints];

    // Get triangle index that will be incise
    // - Creating direction of incision
    const sofa::type::Vec3 dir = startBody.point - the_point;
    // - looking for triangle in this direction
    const auto& shell = startBody.body->getCollisionTopology()->getTrianglesAroundVertex(initialNbPoints);
    const auto triangleIDInShell = sofa::topology::getTriangleIDInDirection(positions, startBody.body->getCollisionTopology()->getTriangles(), shell, initialNbPoints, dir);

    if (triangleIDInShell == sofa::InvalidID)
    {
        msg_error("InciseAlongPathPerformer") << " initial triangle of incision has not been found." ;
        return;
    }
    const auto the_triangle = shell[triangleIDInShell];

    sofa::core::topology::TopologyModifier* topologyModifier;
    startBody.body->getContext()->get(topologyModifier);
    // Handle Removing of topological element (from any type of topology)
    if(topologyModifier)
    {
        topologyChangeManager.incisionCollisionModel(startBody.body,  startBody.indexCollisionElement,  startBody.point,
                firstIncisionBody.body, (unsigned int)the_triangle, the_point,
                snapingValue, snapingBorderValue );
    }

    startBody = firstIncisionBody;
    firstIncisionBody.body = nullptr;

    finishIncision = false; //Incure no second cut
}

InciseAlongPathPerformer::~InciseAlongPathPerformer()
{
    if (secondBody.body)
        secondBody.body= nullptr;

    if (firstBody.body)
        firstBody.body = nullptr;

    if (startBody.body)
        startBody.body = nullptr;

    if (firstIncisionBody.body)
        firstIncisionBody.body = nullptr;

    this->interactor->setBodyPicked(firstIncisionBody);
}

void InciseAlongPathPerformer::draw(const core::visual::VisualParams* vparams)
{
    if (firstBody.body == nullptr) return;

    BodyPicked currentBody=this->interactor->getBodyPicked();

    sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<defaulttype::Vec3Types>* topoGeo;
    firstBody.body->getContext()->get(topoGeo);

    if (!topoGeo)
        return;

    sofa::core::topology::BaseMeshTopology* topoCon;
    firstBody.body->getContext()->get(topoCon);

    if (!topoCon || topoCon->getTriangles().empty())
        return;

    // Output declarations
    sofa::type::vector< sofa::geometry::ElementType> topoPath_list;
    sofa::type::vector<Index> indices_list;
    sofa::type::vector< sofa::type::Vec3 > coords2_list;
    const sofa::type::Vec3 pointA = firstBody.point;
    const sofa::type::Vec3 pointB = currentBody.point;

    sofa::type::vector< sofa::type::Vec3 > positions;
    const bool path_ok = topoGeo->computeIntersectedObjectsList(0, pointA, pointB, firstBody.indexCollisionElement, currentBody.indexCollisionElement, topoPath_list, indices_list, coords2_list);

    if (!path_ok)
        return;

    if (!positions.empty())
        positions.clear();

    positions.resize(topoPath_list.size());

    for (unsigned int i=0; i<topoPath_list.size(); ++i)
    {
        if (topoPath_list[i] == sofa::geometry::ElementType::POINT)
        {
            positions[i] = topoGeo->getPointPosition(indices_list[i]);
        }
        else if (topoPath_list[i] == sofa::geometry::ElementType::EDGE)
        {
            sofa::core::topology::BaseMeshTopology::Edge theEdge = topoCon->getEdge(indices_list[i]);
            const auto AB = topoGeo->getPointPosition(theEdge[1])- topoGeo->getPointPosition(theEdge[0]);
            positions[i] = topoGeo->getPointPosition(theEdge[0]) + AB *coords2_list[i][0];
        }
        else if(topoPath_list[i] == sofa::geometry::ElementType::TRIANGLE)
        {
            sofa::core::topology::BaseMeshTopology::Triangle theTriangle = topoCon->getTriangle(indices_list[i]);

            for (unsigned int j=0; j<3; ++j)
                positions[i] += topoGeo->getPointPosition(theTriangle[j])*coords2_list[i][j];
            positions[i] = positions[i]/3;
        }
    }

    positions[0] = pointA;
    positions[positions.size()-1] = pointB;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();
    constexpr sofa::type::RGBAColor color(0.3f, 0.8f, 0.3f, 1.0f);
    std::vector<sofa::type::Vec3> vertices;
    for (unsigned int i = 1; i<positions.size(); ++i)
    {
        vertices.push_back(sofa::type::Vec3(positions[i-1][0], positions[i-1][1], positions[i-1][2]));
        vertices.push_back(sofa::type::Vec3(positions[i][0], positions[i][1], positions[i][2]));
    }
    vparams->drawTool()->drawLines(vertices,1,color);

}


} // namespace sofa::gui::component::performer
