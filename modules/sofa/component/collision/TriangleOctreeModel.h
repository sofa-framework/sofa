/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_COLLISION_TRIANGLEOCTREEMDEL_H
#define SOFA_COMPONENT_COLLISION_TRIANGLEOCTREEMDEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/TriangleOctree.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class TriangleOctree;
class TriangleModel;

class TriangleOctreeModel:public  TriangleModel
{
public:
    TriangleOctreeModel();
    /*the triangles assiciated to a point*/
    vector<vector<int> > pTri;
    /*the normals for each point*/
    vector<Vector3> pNorms;
    /*the size of the octree cube*/
    int cubeSize;
    /*the first node of the octree*/
    TriangleOctree *octreeRoot;
    //vector < Vector4 > octreeVec;
    void	draw();
    virtual void computeBoundingTree(int maxDepth=0);
    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);
    /*init the octree creation*/
    void buildOctree ();
protected:
    /*used to add a triangle  to the octree*/
    int fillOctree (int t, int d = 0, Vector3 v = Vector3 (0, 0, 0));
};
/*class used to manage the Bounding Box for each triangle*/
class TriangleAABB
{



    double bb[6];

    double m_size;
public:
    double *getAABB ()
    {
        return bb;
    }
    double size ()
    {
        return m_size;
    }
    TriangleAABB (Triangle & t);
};

}				// namespace collision

}				// namespace component

}				// namespace sofa

#endif
