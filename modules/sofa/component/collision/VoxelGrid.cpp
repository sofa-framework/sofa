/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/component/collision/VoxelGrid.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/Sphere.h>
#include <SofaMeshCollision/Triangle.h>
#include <SofaMeshCollision/Line.h>
#include <SofaMeshCollision/Point.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <map>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>

namespace sofa
{

namespace helper
{
void create(VoxelGrid*& obj, simulation::xml::ObjectDescription* arg)
{
    obj = new VoxelGrid(
        Vector3(atof(arg->getAttribute("minx",arg->getAttribute("min","-20.0"))),
                atof(arg->getAttribute("miny",arg->getAttribute("min","-20.0"))),
                atof(arg->getAttribute("minz",arg->getAttribute("min","-20.0")))),
        Vector3(atof(arg->getAttribute("maxx",arg->getAttribute("max","20.0"))),
                atof(arg->getAttribute("maxy",arg->getAttribute("max","20.0"))),
                atof(arg->getAttribute("maxz",arg->getAttribute("max","20.0")))),
        Vector3(atof(arg->getAttribute("nx",arg->getAttribute("n","5.0"))),
                atof(arg->getAttribute("ny",arg->getAttribute("n","5.0"))),
                atof(arg->getAttribute("nz",arg->getAttribute("n","5.0")))),
        atoi(arg->getAttribute("draw","0"))!=0
    );
}

SOFA_DECL_CLASS(VoxelGrid)

Creator<simulation::xml::ObjectFactory, VoxelGrid> VoxelGridClass("VoxelGridDetection");
}
namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace core::objectmodel;

void VoxelGrid::posToIdx (const Vector3& pos, Vector3 &indices)
{
    int i;
    Vector3 nbSubdivisions;

    for (i = 0; i < 3; i++)
        nbSubdivisions[i] = (maxVect[i] - minVect[i]) / step[i];

    indices[0] = (int)((pos[0] - minVect[0]) / step[0]);
    if (indices[0] < 0) indices[0] = 0;
    if (indices[0] >= nbSubdivisions[0]) indices[0] = nbSubdivisions[0] - 1;

    indices[1] = (int)((pos[1] - minVect[1]) / step[1]);
    if (indices[1] < 0) indices[1] = 0;
    if (indices[1] >= nbSubdivisions[1]) indices[1] = nbSubdivisions[1] - 1;

    indices[2] = (int)((pos[2] - minVect[2]) / step[2]);
    if (indices[2] < 0) indices[2] = 0;
    if (indices[2] >= nbSubdivisions[2]) indices[2] = nbSubdivisions[2] - 1;
}

void VoxelGrid::createVoxelGrid (const Vector3& minAxis, const Vector3& maxAxis, const Vector3 &nbSubdivision)
{
    int i, j, k;

    timeStamp = -1;
    minVect = minAxis;
    maxVect = maxAxis;
    //nbSubDiv = nbSubdivision;
    Vector3 minCell, maxCell;

    for (i = 0; i < 3; i++)
        step[i] = (maxVect[i] - minVect[i]) / nbSubdivision[i];

    grid = new GridCell**[(int) nbSubdivision[0]];
    for (i = 0; i < (int)nbSubdivision[0]; i++)
    {
        grid[i] = new GridCell*[(int)nbSubdivision[1]];
        for (j = 0; j < (int)nbSubdivision[1]; j++)
        {
            grid[i][j] = new GridCell[(int)nbSubdivision[2]];
            for (k = 0; k < (int)nbSubdivision[2]; k++)
            {
                minCell[0] = minVect[0] + step[0] * i;
                minCell[1] = minVect[1] + step[1] * j;
                minCell[2] = minVect[2] + step[2] * k;
                maxCell[0] = minCell[0] + step[0];
                maxCell[1] = minCell[1] + step[1];
                maxCell[2] = minCell[2] + step[2];
                grid[i][j][k].setMinMax(minCell, maxCell);
            }
        }
    }
}

static std::map<CollisionModel*,int> timestamps;

int gettimestamp(CollisionModel* cm)
{
    std::map<CollisionModel*,int>::iterator it = timestamps.find(cm);
    if (it == timestamps.end()) return -1;
    else return it->second;
}

int settimestamp(CollisionModel* cm, int t)
{
    timestamps[cm] = t;
    return t;
}

void VoxelGrid::addCollisionModel(CollisionModel *cm)
{
    add(cm,0);
}

void VoxelGrid::add(CollisionModel *cm, int phase)
{
    if (!cm->isStatic() || gettimestamp(cm) < 0)
    {
        //const sofa::helper::vector<CollisionElementIterator>& vectElems = cm->getCollisionElements();
        CollisionElementIterator it = cm->begin();
        CollisionElementIterator itEnd = cm->end();

        const bool proximity  = intersectionMethod->useProximity();
        const double distance = intersectionMethod->getAlarmDistance();

        sofa::helper::vector<CollisionElementIterator> collisionElems;
        Vector3 minBBox, maxBBox;
        Vector3 ijk, lmn;

        for (; it != itEnd; it++)
        {
            minBBox = it->getBBoxMin();
            maxBBox = it->getBBoxMax();
            if (proximity)
            {
                maxBBox[0] += distance;
                maxBBox[1] += distance;
                maxBBox[2] += distance;
            }

            posToIdx (minBBox, ijk);
            posToIdx (maxBBox, lmn);

            core::CollisionModel::clearAllVisits();

            for(int i = (int) ijk[0]; i <= (int)lmn[0]; i++ )
            {
                for(int j = (int) ijk[1] ; j <= (int) lmn[1]; j++ )
                {
                    for(int k = (int) ijk[2] ; k <= (int) lmn[2]; k++ )
                    {
                        //if (grid[i][j][k].timeStamp != timeStamp)
                        grid[i][j][k].eraseAll(timeStamp);
                        grid[i][j][k].add(this, it, collisionElems, phase);
                    }
                }
            }

            // get the collision pair or self collision pair for this model
            sofa::helper::vector<CollisionElementIterator>::const_iterator itCollis = collisionElems.begin();
            sofa::helper::vector<CollisionElementIterator>::const_iterator itCollisEnd = collisionElems.end();

            for (; itCollis != itCollisEnd; ++itCollis)
            {
                //if ((*it)->canCollideWith(*itCollis))
                {
                    cmPairs.push_back(std::pair<CollisionModel*, CollisionModel*>(it->getCollisionModel(), itCollis->getCollisionModel()));
                    elemPairs.push_back(std::pair<CollisionElementIterator, CollisionElementIterator> (it, *itCollis));
                }
            }
            collisionElems.clear();
        }

        if (cm->isStatic())
            settimestamp(cm, 0);
    }
}

void VoxelGrid::addCollisionPair(const std::pair<CollisionModel*, CollisionModel*>& cmPair)
{
    timeLogger = dynamic_cast<simulation::Node*>(getContext());
    if (timeLogger && !timeLogger->getLogTime()) timeLogger=NULL;
    timeInter = 0;

    CollisionModel *cm1 = cmPair.first->getNext(); //cmPair->getCollisionModel(0);
    CollisionModel *cm2 = cmPair.second->getNext(); //getCollisionModel(1);

    if (cm1->isStatic() && cm2->isStatic())
        return;

    if (cm1->empty() || cm2->empty())
        return;

    //const sofa::helper::vector<CollisionElementIterator>& vectElems1 = cm1->getCollisionElements();
    //const sofa::helper::vector<CollisionElementIterator>& vectElems2 = cm2->getCollisionElements();

    if (!intersectionMethod->isSupported(cm1, cm2))
        return;

    if (gettimestamp(cm1) < timeStamp)
        add(cm1,1);
    settimestamp(cm1, timeStamp);
    if (gettimestamp(cm2) < timeStamp)
        add(cm2,1);
    settimestamp(cm2, timeStamp);
    if (timeLogger)
    {
        timeLogger->addTime(timeInter, "collision", intersectionMethod, this);
        timeLogger = NULL;
    }
}

void VoxelGrid::draw(const core::visual::VisualParams* vparams)
{
    if (!bDraw) return;
    Vector3 nbSubdiv;
    int i;

    for (i = 0; i < 3; i++)
        nbSubdiv[i] = (maxVect[i] - minVect[i]) / step[i];

    glDisable(GL_LIGHTING);
    glColor3f (0.0, 0.25, 0.25);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glBegin(GL_QUADS);
    glVertex3d (minVect[0], minVect[1], minVect[2]);
    glVertex3d (minVect[0], maxVect[1], minVect[2]);
    glVertex3d (maxVect[0], maxVect[1], minVect[2]);
    glVertex3d (maxVect[0], minVect[1], minVect[2]);

    glVertex3d (minVect[0], minVect[1], minVect[2]);
    glVertex3d (minVect[0], maxVect[1], minVect[2]);
    glVertex3d (minVect[0], maxVect[1], maxVect[2]);
    glVertex3d (minVect[0], minVect[1], maxVect[2]);

    glVertex3d (maxVect[0], minVect[1], minVect[2]);
    glVertex3d (maxVect[0], maxVect[1], minVect[2]);
    glVertex3d (maxVect[0], maxVect[1], maxVect[2]);
    glVertex3d (maxVect[0], minVect[1], maxVect[2]);

    glVertex3d (minVect[0], maxVect[1], maxVect[2]);
    glVertex3d (maxVect[0], maxVect[1], maxVect[2]);
    glVertex3d (maxVect[0], minVect[1], maxVect[2]);
    glVertex3d (minVect[0], minVect[1], maxVect[2]);

    glEnd();

    for(i = 0; i < (int) nbSubdiv[0]; i++)
    {
        for(int j = 0 ; j < (int) nbSubdiv[1]; j++)
        {
            for(int k = 0 ; k < (int) nbSubdiv[2]; k++)
            {
                grid[i][j][k].draw(vparams,timeStamp);
            }
        }
    }
    sofa::helper::vector<std::pair<CollisionElementIterator, CollisionElementIterator> >::iterator it = elemPairs.begin();
    sofa::helper::vector<std::pair<CollisionElementIterator, CollisionElementIterator> >::iterator itEnd = elemPairs.end();
    if (elemPairs.size() >= 1)
    {
        glDisable(GL_LIGHTING);
        glColor3f(1.0, 0.0, 1.0);
        glLineWidth(3);
        //std::cout << "Size : " << elemPairs.size() << std::endl;
        for (; it != itEnd; ++it)
        {
            it->first->draw(vparams);
            it->second->draw(vparams);
        }
        glLineWidth(1);
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void GridCell::add(VoxelGrid* grid, CollisionElementIterator collisionElem, sofa::helper::vector<CollisionElementIterator> &vectCollis, int phase)
{
    Intersection* intersectionMethod = grid->getIntersectionMethod();
    const bool proximity  = intersectionMethod->useProximity();
    const double distance = intersectionMethod->getAlarmDistance();

    Vector3 minBBox1, maxBBox1;
    //Vector3 minBBox2, maxBBox2;
    minBBox1 = collisionElem.getBBoxMin();
    maxBBox1 = collisionElem.getBBoxMax();

    simulation::Visitor::ctime_t t0 = 0;

    {
        sofa::helper::vector < CollisionElementIterator >	::iterator it	 = collisElems.begin();
        sofa::helper::vector < CollisionElementIterator >	::iterator itEnd = collisElems.end();

        if (proximity)
        {
            minBBox1[0] -= distance;
            minBBox1[1] -= distance;
            minBBox1[2] -= distance;
            maxBBox1[0] += distance;
            maxBBox1[1] += distance;
            maxBBox1[2] += distance;
        }

        for (; it < itEnd; ++it)
        {
            if (!collisionElem.canCollideWith(*it)) continue;
            if (it->visited()) continue;
            it->setVisited();
            const double* minBBox2 = it->getBBoxMin();
            const double* maxBBox2 = it->getBBoxMax();
            if (minBBox1[0] > maxBBox2[0] || minBBox2[0] > maxBBox1[0]
                || minBBox1[1] > maxBBox2[1] || minBBox2[1] > maxBBox1[1]
                || minBBox1[2] > maxBBox2[2] || minBBox2[2] > maxBBox1[2]) continue;

            if (grid->timeLogger) t0 = grid->timeLogger->startTime();
            bool b = intersectionMethod->canIntersect(collisionElem, (*it));
            if (grid->timeLogger) grid->timeInter += grid->timeLogger->startTime() - t0;
            if (b)
                vectCollis.push_back(*it);
        }
    }
    {
        sofa::helper::vector < CollisionElementIterator >::iterator it = collisElemsImmobile[phase].begin();
        sofa::helper::vector < CollisionElementIterator >::iterator itEnd = collisElemsImmobile[phase].end();
        for (; it < itEnd; ++it)
        {
            if (!collisionElem.canCollideWith(*it)) continue;
            if ((*it)->visited()) continue;
            (*it)->setVisited();
            const double* minBBox2 = it->getBBoxMin();
            const double* maxBBox2 = it->getBBoxMax();
            if (minBBox1[0] > maxBBox2[0] || minBBox2[0] > maxBBox1[0]
                || minBBox1[1] > maxBBox2[1] || minBBox2[1] > maxBBox1[1]
                || minBBox1[2] > maxBBox2[2] || minBBox2[2] > maxBBox1[2]) continue;

            if (grid->timeLogger) t0 = grid->timeLogger->startTime();
            bool b = intersectionMethod->canIntersect(collisionElem, (*it));
            if (grid->timeLogger) grid->timeInter += grid->timeLogger->startTime() - t0;
            if (b)
                vectCollis.push_back(*it);
        }
    }
    if (collisionElem->getCollisionModel()->isStatic())
    {
        collisElemsImmobile[phase].push_back(collisionElem);
    }
    else
    {
        collisElems.push_back(collisionElem);
    }
}

void GridCell::eraseAll(int timeStampMethod)
{
    if (timeStampMethod != timeStamp)
    {
        timeStamp = timeStampMethod;
        collisElems.clear();
    }
}

GridCell::GridCell(void)
{
    timeStamp = 0;
}

void GridCell::setMinMax(const Vector3 &minimum, const Vector3 &maximum)
{
    minCell = minimum;
    maxCell = maximum;
}

void GridCell::draw (const core::visual::VisualParams* vparams,int timeStampMethod)
{
    if(timeStampMethod != timeStamp || (collisElems.empty() && collisElemsImmobile[1].empty()))
    {
        return;
    }
    else
    {
        glColor3f (0.0, 1.0, 1.0);
    }

    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glBegin(GL_QUADS);
    glVertex3d (minCell[0], minCell[1], minCell[2]);
    glVertex3d (minCell[0], maxCell[1], minCell[2]);
    glVertex3d (maxCell[0], maxCell[1], minCell[2]);
    glVertex3d (maxCell[0], minCell[1], minCell[2]);

    glVertex3d (minCell[0], minCell[1], minCell[2]);
    glVertex3d (minCell[0], maxCell[1], minCell[2]);
    glVertex3d (minCell[0], maxCell[1], maxCell[2]);
    glVertex3d (minCell[0], minCell[1], maxCell[2]);

    glVertex3d (maxCell[0], minCell[1], minCell[2]);
    glVertex3d (maxCell[0], maxCell[1], minCell[2]);
    glVertex3d (maxCell[0], maxCell[1], maxCell[2]);
    glVertex3d (maxCell[0], minCell[1], maxCell[2]);

    glVertex3d (minCell[0], maxCell[1], maxCell[2]);
    glVertex3d (maxCell[0], maxCell[1], maxCell[2]);
    glVertex3d (maxCell[0], minCell[1], maxCell[2]);
    glVertex3d (minCell[0], minCell[1], maxCell[2]);

    glEnd();
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace collision

} // namespace component

} // namespace sofa

