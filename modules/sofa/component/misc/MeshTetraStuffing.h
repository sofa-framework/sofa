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
#ifndef SOFA_COMPONENT_MISC_MESHTETRASTUFFING_H
#define SOFA_COMPONENT_MISC_MESHTETRASTUFFING_H

#include <string>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

class SOFA_COMPONENT_MISC_API MeshTetraStuffing : public virtual core::objectmodel::BaseObject
{
public:

    typedef defaulttype::Vec3Types::Real Real;
    typedef defaulttype::Vec3Types::Coord Point;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::Tetra Tetra;

    typedef defaulttype::Vec3Types::VecCoord SeqPoints;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetras SeqTetras;

    MeshTetraStuffing();

    virtual ~MeshTetraStuffing();

    virtual void init();

    virtual void draw();

    Data< helper::fixed_array<Point,2> > bbox;
    Data< Real > size;
    Data<SeqPoints> inputPoints;
    Data<SeqTriangles> inputTriangles;
    Data<SeqPoints> outputPoints;
    Data<SeqTetras> outputTetras;
    Data< bool > bDraw;

    Real cellsize;
    int gsize[3];
    Point g0;
    int ph0;
    int hsize[3];
    Point h0;

    enum { EDGESHELL = (6+8) };
    int getEdgePoint2(int p, int e);
    int getEdgeSize2(int e);
    Point getEdgeDir(int e);

    helper::vector<int> pInside;
    helper::vector< helper::fixed_array<Real,EDGESHELL> > eBDist;

    SeqPoints rays;
    SeqPoints intersections;
    SeqPoints insides;

};

} // namespace component

} // namespace sofa

#endif
