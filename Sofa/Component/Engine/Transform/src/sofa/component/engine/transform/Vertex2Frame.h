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
#include <sofa/component/engine/transform/config.h>




#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa::component::engine::transform
{

/**
 * This class gets as inputs the vertices, texCoords, normals and facets of any mesh and returns as output a rigid position
 */
template <class DataTypes>
class Vertex2Frame : public  core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(Vertex2Frame,DataTypes),core::DataEngine);
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::VecCoord VecCoord;

protected:

    Vertex2Frame();

    ~Vertex2Frame() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

protected:
    typename sofa::core::behavior::MechanicalState<DataTypes>::SPtr m_mstate;
    Data< type::vector<CPos> > d_vertices; ///< Vertices of the mesh loaded
    Data< type::vector<sofa::type::Vec2> > d_texCoords; ///< for the moment, we suppose that texCoords is order 2 (2 texCoords for a vertex)
    Data< type::vector<CPos> > d_normals; ///< Normals of the mesh loaded

    Data<VecCoord> d_frames; ///< Frames at output
    Data<bool> d_useNormals; ///< Use normals to compute the orientations; if disabled the direction of the x axisof a vertice is the one from this vertice to the next one
    Data<bool> d_invertNormals; ///< Swap normals

    Data<int> d_rotation; ///< Apply a local rotation on the frames. If 0 a x-axis rotation is applied. If 1 a y-axis rotation is applied, If 2 a z-axis rotation is applied.
    Data<double> d_rotationAngle; ///< Angle rotation

    type::Quat<SReal> computeOrientation(const CPos &xAxis, const CPos &yAxis, const CPos &zAxis);

    static const Real EPSILON;

};

#if !defined(SOFA_COMPONENT_ENGINE_VERTEX2FRAME_CPP)
extern template class SOFA_COMPONENT_ENGINE_TRANSFORM_API Vertex2Frame<defaulttype::Rigid3Types>;
#endif

} //namespace sofa::component::engine::transform
