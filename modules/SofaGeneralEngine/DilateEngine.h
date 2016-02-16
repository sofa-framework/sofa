/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
 * This class dilates the positions of one DataFields into new positions after applying a dilateation
This dilateation can be either translation, rotation, scale
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

    ~DilateEngine() {}

    void init();

    void reinit();

    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const DilateEngine<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    Data<VecCoord> f_inputX; ///< input position
    Data<VecCoord> f_outputX; ///< ouput position
    Data<SeqTriangles> f_triangles; ///< input triangles
    Data<SeqQuads> f_quads; ///< input quads
    Data<VecCoord> f_normals; ///< ouput normals
    Data<helper::vector<Real> > f_thickness;
    Data<Real> f_distance; ///< distance to move the points (positive for dilatation, negative for erosion)
    Data<Real> f_minThickness; ///< minimal thickness to enforce
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_DILATEENGINE_CPP)

//#ifndef SOFA_FLOAT
extern template class SOFA_ENGINE_API DilateEngine<defaulttype::Vec3dTypes>;
//#endif //SOFA_FLOAT
//#ifndef SOFA_DOUBLE
//extern template class SOFA_ENGINE_API DilateEngine<defaulttype::Vec3fTypes>;
//#endif //SOFA_DOUBLE
//extern template class SOFA_ENGINE_API DilateEngine<defaulttype::ExtVec3fTypes>;
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
