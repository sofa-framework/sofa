/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaMeshCollision/TriangleModel.inl>
#include <SofaGeneralMeshCollision/TriangleOctreeModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/Triangle.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/system/thread/CTime.h>

#include <cmath>
#include <sofa/helper/system/gl.h>

namespace sofa
{

namespace component
{

namespace topology
{
typedef core::topology::BaseMeshTopology::Triangle	Triangle;
}

namespace collision
{

using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

int TriangleOctreeModelClass =	core::RegisterObject ("collision model using a triangular mesh mapped to an Octree").add <	TriangleOctreeModel > ().addAlias ("TriangleOctree");

TriangleOctreeModel::TriangleOctreeModel ()
{
}

void TriangleOctreeModel::draw (const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    TriangleModel::draw(vparams);
    if (isActive () && vparams->displayFlags().getShowCollisionModels ())
    {
        if (vparams->displayFlags().getShowWireFrame ())
            glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);

        glEnable (GL_LIGHTING);
        //Enable<GL_BLEND> blending;
        //glLightModeli(GL  _LIGHT_MODEL_TWO_SIDE, GL_TRUE);

        glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, getColor4f());
        static const float emissive[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static const float specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive);
        glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular);
        glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, 20);
        if(octreeRoot)
            octreeRoot->draw(vparams);

        glColor3f (1.0f, 1.0f, 1.0f);
        glDisable (GL_LIGHTING);
        if (vparams->displayFlags().getShowWireFrame ())
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    }
#endif /* SOFA_NO_OPENGL */
}

void TriangleOctreeModel::computeBoundingTree(int maxDepth)
{
    const helper::vector<topology::Triangle>& tri = *triangles;
    if(octreeRoot)
    {
        delete octreeRoot;
        octreeRoot=NULL;
    }

    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();

    if (!isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile
    int size2=mstate->getSize();
    pNorms.resize(size2);
    for(int i=0; i<size2; i++)
    {
        pNorms[i]=defaulttype::Vector3(0,0,0);
    }
    defaulttype::Vector3 minElem, maxElem;
    maxElem[0]=minElem[0]=mstate->read(core::ConstVecCoordId::position())->getValue()[0][0];
    maxElem[1]=minElem[1]=mstate->read(core::ConstVecCoordId::position())->getValue()[0][1];
    maxElem[2]=minElem[2]=mstate->read(core::ConstVecCoordId::position())->getValue()[0][2];

    cubeModel->resize(1);  // size = number of triangles
    for (int i=1; i<size; i++)
    {
        Triangle t(this,i);
        pNorms[tri[i][0]]+=t.n();
        pNorms[tri[i][1]]+=t.n();
        pNorms[tri[i][2]]+=t.n();
        const defaulttype::Vector3* pt[3];
        pt[0] = &t.p1();
        pt[1] = &t.p2();
        pt[2] = &t.p3();
        t.n() = cross(*pt[1]-*pt[0],*pt[2]-*pt[0]);
        t.n().normalize();

        for (int p=0; p<3; p++)
        {


            for(int c=0; c<3; c++)
            {
                if ((*pt[p])[c] > maxElem[c]) maxElem[c] = (*pt[p])[c];
                if ((*pt[p])[c] < minElem[c]) minElem[c] = (*pt[p])[c];

            }
        }

    }

    cubeModel->setParentOf(0, minElem, maxElem); // define the bounding box of the current triangle
    cubeModel->computeBoundingTree(maxDepth);
    for(int i=0; i<size2; i++)
    {
        pNorms[i].normalize();
    }
#if 0
    if(!pTri.size())
    {
        /*creates the list of triangles that are associated to a point*/
        pTri.resize(size2);
        for(int i=0; i<size; i++)
        {
            pTri[tri[i][0]].push_back(i);

            pTri[tri[i][1]].push_back(i);
            pTri[tri[i][2]].push_back(i);
        }
    }
#endif

}

void TriangleOctreeModel::computeContinuousBoundingTree(double/* dt*/, int maxDepth)
{
    computeBoundingTree(maxDepth);
}

void TriangleOctreeModel::buildOctree()
{
    this->octreeTriangles = &this->getTriangles();
    this->octreePos = &this->getX();
    TriangleOctreeRoot::buildOctree();
}

} // namespace collision

} // namespace component

} // namespace sofa
