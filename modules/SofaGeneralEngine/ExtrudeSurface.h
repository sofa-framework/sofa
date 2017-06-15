/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_ENGINE_EXTRUDESURFACE_H
#define SOFA_COMPONENT_ENGINE_EXTRUDESURFACE_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class extrudes a surface
 */
template <class DataTypes>
class ExtrudeSurface : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ExtrudeSurface,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::Vec<3,Real> Vec3;

protected:

    ExtrudeSurface();

    ~ExtrudeSurface() {}
public:
    void init();

    void reinit();

    void update();

    void draw(const core::visual::VisualParams* vparams);

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const ExtrudeSurface<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
    bool initialized;
    Data<bool> isVisible;
    Data<Real> heightFactor;
    Data< helper::vector<sofa::core::topology::BaseMeshTopology::Triangle> > f_triangles;
    Data<VecCoord> f_extrusionVertices;
    Data<VecCoord> f_surfaceVertices;
    Data< helper::vector<sofa::core::topology::BaseMeshTopology::Triangle> > f_extrusionTriangles;
    Data< helper::vector<sofa::core::topology::BaseMeshTopology::TriangleID> > f_surfaceTriangles;


};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_EXTRUDESURFACE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API ExtrudeSurface<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API ExtrudeSurface<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
