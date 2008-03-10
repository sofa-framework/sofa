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
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/system/gl.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Point)

int PointMeshModelClass = core::RegisterObject("Collision model which represents a set of points")
        .add< PointMeshModel >()
        .addAlias("Point")
        .addAlias("PointMesh")
        .addAlias("PointModel")
        ;

int PointSetModelClass = core::RegisterObject("Collision model which represents a set of points")
        .add< PointSetModel >()
        .addAlias("PointSet")
        ;

PointModel::PointModel()
    : mstate(NULL)
{
}

PointMeshModel::PointMeshModel()
    : mesh(NULL)
{
}

PointSetModel::PointSetModel()
    : mesh(NULL)
{
}

void PointModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
}

void PointModel::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::componentmodel::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());

    if (mstate==NULL)
    {
        std::cerr << "ERROR: PointModel requires a Vec3 Mechanical Model.\n";
        return;
    }

    const int npoints = mstate->getX()->size();
    resize(npoints);
}

void PointMeshModel::init()
{
    PointModel::init();
    const int npoints = mstate->getX()->size();

    // If the CollisionDetection Method uses the filtration method based on cones
    if (this->isFiltered())
    {
        topology::MeshTopology *mesh = dynamic_cast< topology::MeshTopology* > (getContext()->getTopology());

        if (mesh != NULL)
        {
            // Line neighborhood construction
            const int nLines = mesh->getNbLines();
            if (nLines != 0)
            {
                lineNeighbors.resize(npoints);

                for (int i=0; i<npoints; i++)
                {
                    lineNeighbors[i].clear();
                    Point p(this,i);

                    const Vector3& pt = p.p();

                    for (int j=0; j<nLines; j++)
                    {
                        topology::MeshTopology::Line idx = mesh->getLine(j);
                        Vector3 a = (*mstate->getX())[idx[0]];
                        Vector3 b = (*mstate->getX())[idx[1]];

                        if (a == pt)
                            lineNeighbors[i].push_back(idx[1]);
                        else if (b == pt)
                            lineNeighbors[i].push_back(idx[0]);
                    }
                }
            }

            // Triangles neighborhood construction
            const int nTriangles = mesh->getNbTriangles();
            if (nTriangles != 0)
            {
                triangleNeighbors.resize(npoints);

                for (int i=0; i<npoints; i++)
                {
                    triangleNeighbors[i].clear();
                    Point p(this,i);

                    const Vector3& pt = p.p();

                    for (int j=0; j<nTriangles; j++)
                    {
                        topology::MeshTopology::Triangle t = mesh->getTriangle(j);
                        Vector3 a = (*mstate->getX())[t[0]];
                        Vector3 b = (*mstate->getX())[t[1]];
                        Vector3 c = (*mstate->getX())[t[2]];

                        if (a == pt)
                        {
                            //	std::cout << "Point " << pt << " voisin du triangle " << pt << ", " << t[1] << ", " << t[2] << std::endl;
                            triangleNeighbors[i].push_back(std::make_pair(t[1], t[2]));
                        }
                        else if (b == pt)
                        {
                            //	std::cout << "Point " << pt << " voisin du triangle " << pt << ", " << t[2] << ", " << t[0] << std::endl;
                            triangleNeighbors[i].push_back(std::make_pair(t[2], t[0]));
                        }
                        else if (c == pt)
                        {
                            //	std::cout << "Point " << pt << " voisin du triangle " << pt << ", " << t[0] << ", " << t[1] << std::endl;
                            triangleNeighbors[i].push_back(std::make_pair(t[0], t[1]));
                        }
                    }
                }
            }
        }
    }
}
void PointSetModel::init()
{
    PointModel::init();
}

void PointModel::draw(int index)
{
    Point t(this,index);
    glBegin(GL_POINTS);
    glVertex3dv(t.p().ptr());
    glEnd();
}

void PointModel::draw()
{
    if (getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glDisable(GL_LIGHTING);
        glPointSize(3);
        glColor4fv(getColor4f());

        for (int i=0; i<size; i++)
        {
            draw(i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        glPointSize(1);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

bool PointModel::canCollideWithElement(int index, CollisionModel* model2, int index2)
{
    if (!this->bSelfCollision.getValue()) return true;
    if (this->getContext() != model2->getContext()) return true;
    if (model2 == this)
    {
        //std::cout << "point self test "<<index<<" - "<<index2<<std::endl;
        return index < index2-1; // || index > index2+1;
    }
    else
        return model2->canCollideWithElement(index2, this, index);
}

void PointModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
    }
    if (updated) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    cubeModel->resize(size);
    if (!empty())
    {
        //VecCoord& x = *mstate->getX();
        for (int i=0; i<size; i++)
        {
            Point p(this,i);
            const Vector3& pt = p.p();
            cubeModel->setParentOf(i, pt, pt);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void PointModel::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
    }
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        //VecCoord& x = *mstate->getX();
        //VecDeriv& v = *mstate->getV();
        for (int i=0; i<size; i++)
        {
            Point p(this,i);
            const Vector3& pt = p.p();
            const Vector3 ptv = pt + p.v()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt[c];
                maxElem[c] = pt[c];
                if (ptv[c] > maxElem[c]) maxElem[c] = ptv[c];
                else if (ptv[c] < minElem[c]) minElem[c] = ptv[c];
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void Point::getLineNeighbors(std::vector< Vector3> &nV) const
{
    std::vector<int> v = model->lineNeighbors[index];

    std::vector<int>::iterator it = v.begin();
    std::vector<int>::iterator itEnd = v.end();

    nV.clear();
//	nV.resize(v.size());

    while(it != itEnd)
    {
        nV.push_back((*model->mstate->getX())[*it]);
        ++it;
    }
}

void Point::getTriangleNeighbors(std::vector< std::pair< Vector3, Vector3 > > &nV) const
{
    if (!model->triangleNeighbors.size())
        return;

    std::vector< std::pair<int, int> > v = model->triangleNeighbors[index];

    std::vector< std::pair<int, int> >::iterator it = v.begin();
    std::vector< std::pair<int, int> >::iterator itEnd = v.end();

    nV.clear();
//	nV.resize(v.size());

    while(it != itEnd)
    {
        nV.push_back(std::make_pair((*model->mstate->getX())[it->first], (*model->mstate->getX())[it->second]));
        ++it;
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

