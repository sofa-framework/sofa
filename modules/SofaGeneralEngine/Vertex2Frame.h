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
#ifndef SOFA_COMPONENT_ENGINE_VERTEX2FRAME_H
#define SOFA_COMPONENT_ENGINE_VERTEX2FRAME_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif


#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa
{

namespace component
{

namespace engine
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

    ~Vertex2Frame() {}
public:
    void init() override;

    void reinit() override;

    void update() override;

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const Vertex2Frame<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    typename sofa::core::behavior::MechanicalState<DataTypes>::SPtr m_mstate;
    Data< helper::vector<CPos> > d_vertices;
    Data< helper::vector<sofa::defaulttype::Vector2> > d_texCoords; // for the moment, we suppose that texCoords is order 2 (2 texCoords for a vertex)
    Data< helper::vector<CPos> > d_normals;

    Data<VecCoord> d_frames;
    Data<bool> d_useNormals;
    Data<bool> d_invertNormals;

    Data<int> d_rotation;
    Data<double> d_rotationAngle;

    defaulttype::Quat computeOrientation(const CPos &xAxis, const CPos &yAxis, const CPos &zAxis);

    static const Real EPSILON;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_VERTEX2FRAME_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API Vertex2Frame<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API Vertex2Frame<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
