#include "VoxelGrid.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Common/FnDispatcher.h"
#include "Common/ObjectFactory.h"

#include <map>

/* for debugging the collision method */
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glut.h>

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Collision;

void create(VoxelGrid*& obj, ObjectDescription* arg)
{
    obj = new VoxelGrid(arg->getName(),
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

Creator<ObjectFactory, VoxelGrid> VoxelGridClass("VoxelGridDetection");

using namespace Abstract;

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
        const std::vector<CollisionElement*>& vectElems = cm->getCollisionElements();
        std::vector<CollisionElement*>::const_iterator it = vectElems.begin();
        std::vector<CollisionElement*>::const_iterator itEnd = vectElems.end();

        std::set<CollisionElement*> collisionElems;
        Vector3 minBBox, maxBBox;
        Vector3 ijk, lmn;
        bool collisionDetected = false;

        for (; it != itEnd; it++)
        {
            (*it)->getBBox(minBBox, maxBBox);
            posToIdx (minBBox, ijk);
            posToIdx (maxBBox, lmn);

            for(int i = (int) ijk[0]; i <= (int)lmn[0]; i++ )
            {
                for(int j = (int) ijk[1] ; j <= (int) lmn[1]; j++ )
                {
                    for(int k = (int) ijk[2] ; k <= (int) lmn[2]; k++ )
                    {
                        //if (grid[i][j][k].timeStamp != timeStamp)
                        grid[i][j][k].eraseAll(timeStamp);
                        grid[i][j][k].add((*it), collisionElems,phase);
                    }
                }
            }

            // get the collision pair or self collision pair for this sphere
            std::set<CollisionElement*>::iterator itCollis = collisionElems.begin();
            std::set<CollisionElement*>::iterator itCollisEnd = collisionElems.end();

            for (; itCollis != itCollisEnd; itCollis++)
            {
                //if ((*it)->canCollideWith(*itCollis))
                {
                    collisionDetected = true;
                    cmPairs.push_back(std::pair<CollisionModel*, CollisionModel*>((*it)->getCollisionModel(), (*itCollis)->getCollisionModel()));
                    elemPairs.push_back(std::pair<CollisionElement*, CollisionElement*> (*it, *itCollis));

                    if ((cm == (*it)->getCollisionModel()) || (cm == (*itCollis)->getCollisionModel()))
                    {
                        removeCmNoCollision ((*itCollis)->getCollisionModel());
                    }
                }
            }
            collisionElems.clear();
        }

        if (!collisionDetected)
            addNoCollisionDetect(cm);
        else
            removeCmNoCollision (cm);

        if (cm->isStatic())
            settimestamp(cm, 0);
    }
}

void VoxelGrid::addCollisionPair(const std::pair<CollisionModel*, CollisionModel*>& cmPair)
{
    CollisionModel *cm1 = cmPair.first->getNext(); //cmPair->getCollisionModel(0);
    CollisionModel *cm2 = cmPair.second->getNext(); //getCollisionModel(1);
    if (gettimestamp(cm1) < timeStamp)
        add(cm1,1);
    settimestamp(cm1, timeStamp);
    if (gettimestamp(cm2) < timeStamp)
        add(cm2,1);
    settimestamp(cm2, timeStamp);
}

void VoxelGrid::draw()
{
    //if (!Scene::getInstance()->getShowCollisionModels()) return;
    if (!bDraw) return;
    Vector3 nbSubdiv;
    int i;

    for (i = 0; i < 3; i++)
        nbSubdiv[i] = (maxVect[i] - minVect[i]) / step[i];

    glColor3f (0.25, 0.0, 0.25);

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
                grid[i][j][k].draw(timeStamp);
            }
        }
    }
    std::vector<std::pair<CollisionElement*, CollisionElement*> >::iterator it = elemPairs.begin();
    std::vector<std::pair<CollisionElement*, CollisionElement*> >::iterator itEnd = elemPairs.end();
    if (elemPairs.size() >= 1)
    {
        glDisable(GL_LIGHTING);
        glColor3f(1.0, 0.0, 1.0);
        glLineWidth(3);
        //std::cout << "Size : " << elemPairs.size() << std::endl;
        for (; it != itEnd; it++)
        {
            //std::cout<<*((*it)->getCollisionElement(0)->getSphere())<<std::endl;
            Sphere *s;
            s = dynamic_cast<Sphere*>(it->first);
            if (s!=NULL) s->draw();
            s = dynamic_cast<Sphere*>(it->second);
            if (s!=NULL) s->draw();
            Triangle *t;
            t = dynamic_cast<Triangle*>(it->first);
            if (t!=NULL) t->draw();
            t = dynamic_cast<Triangle*>(it->second);
            if (t!=NULL) t->draw();
            /* Sphere *sph = (*it)->first->getSphere();
            Sphere *sph1 = (*it)->second->getSphere();
            glPushMatrix();
            glTranslated (sph->center.x, sph->center.y, sph->center.z);
            glutSolidSphere(sph->radius, 10, 10);
            glPopMatrix();
            glPushMatrix();
            glTranslated (sph1->center.x, sph1->center.y, sph1->center.z);
            glutSolidSphere(sph1->radius, 10, 10);
            glPopMatrix(); */
            //(*it)->getCollisionElement(0)->getSphere()->draw();
            //(*it)->getCollisionElement(1)->getSphere()->draw();
        }
        glLineWidth(1);
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void GridCell::add(CollisionElement *collisionElem, std::set<CollisionElement*> &vectCollis, int phase)
{
    FnCollisionDetection *intersectionDis = FnCollisionDetection::getInstance();

    std::vector < CollisionElement* >	::iterator it	 = collisElems.begin();
    std::vector < CollisionElement* >	::iterator itEnd = collisElems.end();

    Vector3 minBBox1, maxBBox1;
    //Vector3 minBBox2, maxBBox2;
    //collisionElem->getBBox(minBBox1, maxBBox1);

    for (; it < itEnd; it++)
    {
        if (!(*collisionElem).canCollideWith(*it)) continue;
        //(*it)->getBBox(minBBox2, maxBBox2);
        //if (minBBox1[0] > maxBBox2[0] || minBBox2[0] > maxBBox1[0]
        // || minBBox1[1] > maxBBox2[1] || minBBox2[1] > maxBBox1[1]
        // || minBBox1[2] > maxBBox2[2] || minBBox2[2] > maxBBox1[2]) continue;

        if (intersectionDis->intersection(*collisionElem, *(*it)))
            vectCollis.insert(*it);
    }

    std::vector < CollisionElement* >::iterator itImmo = collisElemsImmobile[phase].begin();
    std::vector < CollisionElement* >::iterator itImmoEnd = collisElemsImmobile[phase].end();

    for (; itImmo < itImmoEnd; itImmo++)
    {
        if (!(*collisionElem).canCollideWith(*itImmo)) continue;
        //(*itImmo)->getBBox(minBBox2, maxBBox2);
        //if (minBBox1[0] > maxBBox2[0] || minBBox2[0] > maxBBox1[0]
        // || minBBox1[1] > maxBBox2[1] || minBBox2[1] > maxBBox1[1]
        // || minBBox1[2] > maxBBox2[2] || minBBox2[2] > maxBBox1[2]) continue;
        if (intersectionDis->intersection(*collisionElem, *(*itImmo)))
            vectCollis.insert(*itImmo);
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

void GridCell::draw (int timeStampMethod)
{
    if (timeStampMethod != timeStamp || (collisElems.empty() && collisElemsImmobile[1].empty()))
    {
        return;
        glColor3f (0.25, 0.0, 0.25);
    }
    else
    {
        glColor3f (1.0, 0.0, 1.0);
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

} // namespace Components

} // namespace Sofa
