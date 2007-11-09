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
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/TriangleOctreeModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/Triangle.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/ObjectFactory.h>
#include <vector>
#include <sofa/helper/system/thread/CTime.h>

#include <cmath>
#include <GL/gl.h>
#include <GL/glut.h>


namespace sofa
{

namespace component
{

namespace collision
{

using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;



TriangleAABB::TriangleAABB (Triangle & t)
{
    for (int i = 0; i < 3; i++)
    {

        bb[i * 2] = bb_min3 (t.p1 ()[i], t.p2 ()[i], t.p3 ()[i]);
        bb[(i * 2) + 1] = bb_max3 (t.p1 ()[i], t.p2 ()[i], t.p3 ()[i]);

        m_size =
            bb_max3 (fabs (bb[1] - bb[0]), fabs (bb[3] - bb[2]),
                    fabs (bb[5] - bb[4]));


    }
}




int TriangleOctreeModelClass =	core::RegisterObject ("collision model using a triangular mesh mapped to an Octree").add <	TriangleOctreeModel > ().addAlias ("TriangleOctree");

TriangleOctreeModel::TriangleOctreeModel ()
{
    TriangleModel();
    octreeRoot = NULL;
    cubeSize = CUBE_SIZE;
}



void TriangleOctreeModel::draw ()
{

    if (isActive () && getContext ()->getShowCollisionModels ())
    {
        if (getContext ()->getShowWireFrame ())
            glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);

        glEnable (GL_LIGHTING);
        //Enable<GL_BLEND> blending;
        //glLightModeli(GL  _LIGHT_MODEL_TWO_SIDE, GL_TRUE);

        static const float color[4] = { 1.0f, 0.2f, 0.0f, 1.0f };
        static const float colorStatic[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
        if (isStatic ())
            glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
                    colorStatic);
        else
            glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
        static const float emissive[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        static const float specular[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive);
        glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular);
        glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, 20);
        if(octreeRoot)
            octreeRoot->draw ();

        glColor3f (1.0f, 1.0f, 1.0f);
        glDisable (GL_LIGHTING);
        if (getContext ()->getShowWireFrame ())
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    }
}

int TriangleOctreeModel::fillOctree (int tId, int /*d*/, Vector3 /*v*/)
{


    Vector3 center;
    Triangle t (this, tId);
    Vector3 corner (-cubeSize, -cubeSize, -cubeSize);

    TriangleAABB aabb (t);
    double *bb = aabb.getAABB ();
    /*Computes the depth of the bounding box in a octree

     */
    int d1 = (int) log2 ((CUBE_SIZE * 2) / aabb.size ());
    /*computes the size of the octree box that can store the bounding box */
    int divs = (1 << (d1));
    double inc = (double) (2 * CUBE_SIZE) / divs;
    if (bb[0] >= -CUBE_SIZE && bb[2] >= -CUBE_SIZE && bb[4] >= -CUBE_SIZE
        && bb[1] <= CUBE_SIZE && bb[3] <= CUBE_SIZE && bb[5] <= CUBE_SIZE)
        for (double x1 =
                (((int)((bb[0] + CUBE_SIZE) / inc)) * inc - CUBE_SIZE);
                x1 <= bb[1]; x1 += inc)
        {

            for (double y1 =
                    ((int)((bb[2] + CUBE_SIZE) / inc)) * inc - CUBE_SIZE;
                    y1 <= bb[3]; y1 += inc)
            {


                for (double z1 =
                        ((int)((bb[4] + CUBE_SIZE) / inc)) * inc - CUBE_SIZE;
                        z1 <= bb[5]; z1 += inc)
                {
                    octreeRoot->insert (x1, y1, z1, inc, tId);

                }
            }
        }
    return 0;

}
void TriangleOctreeModel::computeBoundingTree(int maxDepth)
{
    if(octreeRoot)
    {
        delete octreeRoot;
        octreeRoot=NULL;
    }

    CubeModel* cubeModel = createPrevious<CubeModel>();
    updateFromTopology();

    if (isStatic() && !cubeModel->empty()) return; // No need to recompute BBox if immobile
    int size2=mstate->getX()->size();
    pNorms.resize(size2);
    for(int i=0; i<size2; i++)
    {
        pNorms[i]=Vector3(0,0,0);
    }
    Vector3 minElem, maxElem;
    maxElem[0]=minElem[0]=(*mstate->getX())[0][0];
    maxElem[1]=minElem[1]=(*mstate->getX())[0][1];
    maxElem[2]=minElem[2]=(*mstate->getX())[0][2];

    cubeModel->resize(1);  // size = number of triangles
    for (int i=1; i<size; i++)
    {
        Triangle t(this,i);
        pNorms[elems[i].i1]+=t.n();
        pNorms[elems[i].i2]+=t.n();
        pNorms[elems[i].i3]+=t.n();
        const Vector3* pt[3];
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
    if(!pTri.size())
    {
        /*creates the list of triangles that are associated to a point*/
        pTri.resize(size2);
        for(int i=0; i<size; i++)
        {
            pTri[elems[i].i1].push_back(i);

            pTri[elems[i].i2].push_back(i);
            pTri[elems[i].i3].push_back(i);
        }
    }
}

void TriangleOctreeModel::computeContinuousBoundingTree(double/* dt*/, int maxDepth)
{
    computeBoundingTree(maxDepth);

}
void TriangleOctreeModel::buildOctree ()
{
    octreeRoot = new TriangleOctree (this);

    /*for each triangle add it to the octree*/
    for (size_t i = 0; i < elems.size (); i++)
    {

        fillOctree (i);

    }
}

}				// namespace collision

}				// namespace component

}				// namespace sofa
