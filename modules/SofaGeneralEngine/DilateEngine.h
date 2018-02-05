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
#ifndef SOFA_COMPONENT_ENGINE_DILATEENGINE_H
#define SOFA_COMPONENT_ENGINE_DILATEENGINE_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class dilates a given mesh by moving vertices along their normal.
 */
template <class DataTypes>
class DilateEngine : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DilateEngine,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef typename SeqTriangles::value_type Triangle;
    typedef typename SeqQuads::value_type Quad;

public:

    DilateEngine();

    virtual ~DilateEngine() {}

    virtual void init() override;
    virtual void bwdInit() override;
    virtual void reinit() override;
    virtual void update() override;

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const DilateEngine<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    Data<VecCoord> d_inputX; ///< input position
    Data<VecCoord> d_outputX; ///< ouput position
    Data<SeqTriangles> d_triangles; ///< input triangles
    Data<SeqQuads> d_quads; ///< input quads
    Data<VecCoord> d_normals; ///< ouput normals
    Data<helper::vector<Real> > d_thickness;
    Data<Real> d_distance; ///< distance to move the points (positive for dilatation, negative for erosion)
    Data<Real> d_minThickness; ///< minimal thickness to enforce
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_DILATEENGINE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API DilateEngine<defaulttype::Vec3dTypes>;
#endif
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
