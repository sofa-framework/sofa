/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/engine/generate/config.h>



#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::engine::generate
{

/**
 * This class dilates the positions of one DataFields into new positions after applying a dilateation
This dilateation can be either translation, rotation, scale
 */
template <class DataTypes>
class GenerateCylinder : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(GenerateCylinder,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef typename sofa::core::topology::Topology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef typename SeqTetrahedra::value_type Tetrahedron;
    typedef typename SeqTriangles::value_type Triangle;

public:

    GenerateCylinder();

    ~GenerateCylinder() override {}

    void init() override;

    void reinit() override;

    void doUpdate() override;

public:
    Data<VecCoord> f_outputTetrahedraPositions; ///< ouput tetrahedra position
	Data<VecCoord> f_outputTrianglesPositions; ///< ouput triangle positions
    Data<SeqTetrahedra> f_tetrahedra; ///< output tetrahedra
	Data<SeqTriangles> f_triangles; ///< output triangles
	Data<sofa::type::vector<Real> > f_bezierTriangleWeight; ///<  output weight for rational Bezier triangles
	Data<sofa::type::vector<bool> > f_isBezierTriangleRational; ///<  for each Bezier triangle indicates if it is rational or integral
    Data<size_t> f_bezierTriangleDegree; ///< degree of Bezier triangles
	Data<sofa::type::vector<Real> > f_bezierTetrahedronWeight; ///<  output weight for rational Bezier triangles
    Data<sofa::type::vector<bool> > f_isBezierTetrahedronRational; ///<  for each Bezier tetrahedron indicates if it is rational
	Data<size_t> f_bezierTetrahedronDegree; ///< degree of Bezier tetrahedron
    Data<Real > f_radius; ///< radius of cylinder 
	Data<Real > f_height; ///< height of cylinder
    Data<Coord> f_origin; ///< origin
    Data<bool> f_openSurface; ///< if the triangulated surface is open or not
    Data<size_t> f_resolutionCircumferential; ///< number of points in the circumferential direction
    Data<size_t> f_resolutionRadial; ///< number of points in the radial  direction
    Data<size_t> f_resolutionHeight; ///< number of points in the height direction
};


#if !defined(SOFA_COMPONENT_ENGINE_GENERATECYLINDER_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API GenerateCylinder<defaulttype::Vec3Types>;

#endif

} //namespace sofa::component::engine::generate
