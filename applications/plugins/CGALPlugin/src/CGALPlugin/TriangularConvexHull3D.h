/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#pragma once

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/Data.h>

namespace cgal
{

using namespace sofa::core::objectmodel;

template <class DataTypes>
class TriangularConvexHull3D : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularConvexHull3D,DataTypes),sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::Tetra Tetra;

    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;


public:
    TriangularConvexHull3D();
    virtual ~TriangularConvexHull3D() { };

    void init() override;
    void reinit() override;

    void doUpdate() override;

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const TriangularConvexHull3D<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Inputs
    sofa::core::objectmodel::Data<VecCoord> f_X0; ///< Rest position coordinates of the degrees of freedom

    //Outputs
    sofa::core::objectmodel::Data<VecCoord> f_newX0; ///< New Rest position coordinates
    sofa::core::objectmodel::Data<SeqTriangles> f_triangles; ///< List of triangles
};

#if !defined(CGALPLUGIN_TRIANGULARCONVEXHULL3D_CPP)
extern template class SOFA_CGALPLUGIN_API TriangularConvexHull3D<defaulttype::Vec3Types>;
#endif

} //cgal

