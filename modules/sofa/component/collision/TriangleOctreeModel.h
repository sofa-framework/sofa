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
#ifndef SOFA_COMPONENT_COLLISION_TRIANGLEOCTREEMODEL_H
#define SOFA_COMPONENT_COLLISION_TRIANGLEOCTREEMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/TriangleOctree.h>


/*This fixed size must be changed*/
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

    int cubeSize;
    TriangleOctree *octreeRoot;
    vector < Vector4 > octreeVec;
    void	draw();
    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0)
    {
        if(octreeRoot)
        {
            delete octreeRoot;
            octreeRoot=NULL;
        }
        //buildOctree();
        this->TriangleModel::computeContinuousBoundingTree(0);
    }
    void buildOctree ();
    int fillOctree (int t, int d = 0, Vector3 v = Vector3 (0, 0, 0));
};
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
