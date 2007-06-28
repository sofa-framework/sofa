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
        //std::cerr<<"p1:"<<t.p1 ()[i]<<" p2:"<<t.p2 ()[i]<<" p3:"<<  t.p3 ()[i]<<std::endl;
        //std::cerr<<"min:"<<bb[i*2]<<" max:"<<bb[(i*2)+1] <<std::endl;
    }
    m_size =
        bb_max3 (fabs (bb[1] - bb[0]), fabs (bb[3] - bb[2]),
                fabs (bb[5] - bb[4]));
    if (!m_size)
    {
        std::cerr << "zero:" << t.p1 () << ", " << t.p2 () << ", " << t.
                p3 () << std::endl;
        std::
        cerr << "values1:" << bb[1] << " " << bb[0] << "," << bb[3] <<
                " " << bb[2] << "," << bb[5] << " " << bb[4] << std::endl;
        std::cerr << "values:" << abs (bb[1] -
                bb[0]) << "," << abs (bb[3] -
                        bb[2]) << ","
                << abs (bb[5] - bb[4]) << std::endl;
    }
    //std::cerr<<"size:"<<m_size<<std::endl;

}
//


int TriangleOctreeModelClass =
    core::RegisterObject ("collision model using a triangular mesh").add <
    TriangleOctreeModel > ().addAlias ("Triangle");

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

        octreeRoot->draw ();

        glColor3f (1.0f, 1.0f, 1.0f);
        glDisable (GL_LIGHTING);
        if (getContext ()->getShowWireFrame ())
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    }
}

int TriangleOctreeModel::fillOctree (int tId, int d, Vector3 v)
{


    Vector3 center;
    Vector3 pt[3];
    Triangle t (this, tId);
    double triang[3][3];
    Vector3 corner (-cubeSize, -cubeSize, -cubeSize);
    double normal[3];

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
void TriangleOctreeModel::buildOctree ()
{
    ctime_t t0, t1, t2;
    if (octreeRoot)
    {
        delete octreeRoot;
        octreeRoot = NULL;
    }
    t0 = CTime::getRefTime ();

    octreeRoot = new TriangleOctree (this);
    for (int i = 0; i < elems.size (); i++)
    {

        fillOctree (i);

    }
    t1 = CTime::getRefTime ();

    octreeRoot->traceVolume (20);
    t2 = CTime::getRefTime ();

    std::cerr << "Octree construction:" << (t1 -
            t0) /
            ((double) CTime::getRefTicksPerSec () /
                    1000) << " traceVolume:" << (t2 -
                            t1) /
            ((double) CTime::getRefTicksPerSec () / 1000) << std::endl;
}

}				// namespace collision

}				// namespace component

}				// namespace sofa
