/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/*
 * CuboidMesh.h
 *
 *  Created on: 12 sep. 2011
 *      Author: Yiyi
 */

#ifndef CGALPLUGIN_CUBOIDMESH_H
#define CGALPLUGIN_CUBOIDMESH_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/gl/template.h>

#include <math.h>
#include   <algorithm>

namespace cgal
{

template <class DataTypes>
class CuboidMesh : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CuboidMesh,DataTypes),sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef sofa::helper::fixed_array<int, 3> Index;
    //    typedef sofa::helper::vector<Real> VecReal;

    //        typedef sofa::core::topology::BaseMeshTopology::PointID PointID;
    //    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    //    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    //    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::Tetra Tetra;
    typedef sofa::core::topology::BaseMeshTopology::Hexa Hexa;


    //    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    //    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    //    typedef sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;


public:
    CuboidMesh();
    virtual ~CuboidMesh() { };

    void init();
    void reinit();

    void update();
    void orientate();
    void draw();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const CuboidMesh<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Inputs
    sofa::core::objectmodel::Data<unsigned> m_debug; ///< for test
    sofa::core::objectmodel::Data<double> m_radius; ///< radius
    sofa::core::objectmodel::Data<double> m_height; ///< height
    sofa::core::objectmodel::Data<int> m_number; ///< number of intervals
    sofa::core::objectmodel::Data<bool> m_convex; ///< make convex boundary
    sofa::core::objectmodel::Data<bool> m_viewPoints; ///< Display Points
    sofa::core::objectmodel::Data<bool> m_viewTetras; ///< Display Tetrahedra

    //Outputs
    sofa::core::objectmodel::Data<VecCoord> m_points; ///< Points
    sofa::core::objectmodel::Data<SeqTetrahedra> m_tetras; ///< Tetrahedra

    //Parameters
    unsigned m_nbVertices, m_nbBdVertices, m_nbCenters, m_nbBdCenters;
    unsigned m_nbTetras_i, m_nbTetras_j, m_nbTetras_k;
    int n, m;
    Real r, h, d, t;
    std::map<Index, unsigned> m_ptID;
    unsigned short debug;


};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(CGALPLUGIN_CUBOIDMESH_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CGALPLUGIN_API CuboidMesh<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_CGALPLUGIN_API CuboidMesh<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} //cgal

#endif /* CGALPLUGIN_CUBOIDMESH_H */
