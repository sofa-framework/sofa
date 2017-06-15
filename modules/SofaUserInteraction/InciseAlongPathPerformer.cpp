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
#include <SofaUserInteraction/InciseAlongPathPerformer.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>

#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace component
{

namespace collision
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
        initialNbTriangles = startBody.body->getMeshTopology()->getNbTriangles();
        initialNbPoints = startBody.body->getMeshTopology()->getNbPoints();
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
        if (firstBody.body == NULL) // first clic
            firstBody=startBody;
        else
        {
            if (firstBody.indexCollisionElement != startBody.indexCollisionElement)
                secondBody=startBody;
        }


        if (firstBody.body == NULL || secondBody.body == NULL) return;

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
        secondBody.body = NULL;

        this->interactor->setBodyPicked(secondBody);
    }
    else
    {

        BodyPicked currentBody=this->interactor->getBodyPicked();
        if (currentBody.body == NULL || startBody.body == NULL) return;

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

        currentBody.body=NULL;
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
    if (firstIncisionBody.body == NULL || startBody.body == NULL)
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
    sofa::helper::vector <unsigned int> triAroundVertex = startBody.body->getMeshTopology()->getTrianglesAroundVertex(initialNbPoints);

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
    sofa::component::container::MechanicalObject<defaulttype::Vec3Types>* MechanicalObject=NULL;
    startBody.body->getContext()->get(MechanicalObject, sofa::core::objectmodel::BaseContext::SearchRoot);
    const sofa::defaulttype::Vector3& the_point = (MechanicalObject->read(core::ConstVecCoordId::position())->getValue())[initialNbPoints];

    // Get triangle index that will be incise
    // - Creating direction of incision
    sofa::defaulttype::Vector3 dir = startBody.point - the_point;
    // - looking for triangle in this direction
    sofa::component::topology::TriangleSetGeometryAlgorithms<defaulttype::Vec3Types>* triangleGeometry;
    startBody.body->getContext()->get(triangleGeometry);
    int the_triangle = triangleGeometry->getTriangleInDirection(initialNbPoints, dir);

    if (the_triangle == -1)
    {
        msg_error("InciseAlongPathPerformer") << " initial triangle of incision has not been found." ;
        return;
    }

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
    firstIncisionBody.body = NULL;

    finishIncision = false; //Incure no second cut
}

InciseAlongPathPerformer::~InciseAlongPathPerformer()
{
    if (secondBody.body)
        secondBody.body= NULL;

    if (firstBody.body)
        firstBody.body = NULL;

    if (startBody.body)
        startBody.body = NULL;

    if (firstIncisionBody.body)
        firstIncisionBody.body = NULL;

    this->interactor->setBodyPicked(firstIncisionBody);
}

void InciseAlongPathPerformer::draw(const core::visual::VisualParams* )
{
#ifndef SOFA_NO_OPENGL
    if (firstBody.body == NULL) return;

    BodyPicked currentBody=this->interactor->getBodyPicked();

    sofa::component::topology::TriangleSetGeometryAlgorithms<defaulttype::Vec3Types>* topoGeo;
    firstBody.body->getContext()->get(topoGeo);

    if (!topoGeo)
        return;

    sofa::component::topology::TriangleSetTopologyContainer* topoCon;
    firstBody.body->getContext()->get(topoCon);

    if (!topoCon)
        return;

    // Output declarations
    sofa::helper::vector< sofa::core::topology::TopologyObjectType> topoPath_list;
    sofa::helper::vector<unsigned int> indices_list;
    sofa::helper::vector< sofa::defaulttype::Vec<3, double> > coords2_list;
    sofa::defaulttype::Vec<3,double> pointA = firstBody.point;
    sofa::defaulttype::Vec<3,double> pointB = currentBody.point;

    sofa::helper::vector< sofa::defaulttype::Vec<3, double> > positions;
    bool path_ok = topoGeo->computeIntersectedObjectsList(0, pointA, pointB, firstBody.indexCollisionElement, currentBody.indexCollisionElement, topoPath_list, indices_list, coords2_list);

    if (!path_ok)
        return;

    if (!positions.empty())
        positions.clear();

    positions.resize(topoPath_list.size());

    for (unsigned int i=0; i<topoPath_list.size(); ++i)
    {
        if (topoPath_list[i] == sofa::core::topology::POINT)
        {
            positions[i] = topoGeo->getPointPosition(indices_list[i]);
        }
        else if (topoPath_list[i] == sofa::core::topology::EDGE)
        {
            sofa::core::topology::BaseMeshTopology::Edge theEdge = topoCon->getEdge(indices_list[i]);
            const sofa::defaulttype::Vec<3, double> AB = topoGeo->getPointPosition(theEdge[1])- topoGeo->getPointPosition(theEdge[0]);
            positions[i] = topoGeo->getPointPosition(theEdge[0]) + AB *coords2_list[i][0];
        }
        else if(topoPath_list[i] == sofa::core::topology::TRIANGLE)
        {
            sofa::core::topology::BaseMeshTopology::Triangle theTriangle = topoCon->getTriangle(indices_list[i]);

            for (unsigned int j=0; j<3; ++j)
                positions[i] += topoGeo->getPointPosition(theTriangle[j])*coords2_list[i][j];
            positions[i] = positions[i]/3;
        }
    }

    positions[0] = pointA;
    positions[positions.size()-1] = pointB;

    glDisable(GL_LIGHTING);
    glColor3f(0.3f,0.8f,0.3f);
    glBegin(GL_LINES);

    for (unsigned int i = 1; i<positions.size(); ++i)
    {
        glVertex3d(positions[i-1][0], positions[i-1][1], positions[i-1][2]);
        glVertex3d(positions[i][0], positions[i][1], positions[i][2]);
    }

    glEnd();
#endif /* SOFA_NO_OPENGL */
}


}
}
}

