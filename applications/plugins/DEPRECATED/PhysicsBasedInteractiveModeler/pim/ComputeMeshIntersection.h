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
#ifndef PLUGINS_PIM_COMPUTEMESHINTERSECTION_H
#define PLUGINS_PIM_COMPUTEMESHINTERSECTION_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/MeshTopology.h>

namespace plugins
{

namespace pim
{

using namespace sofa::core::behavior;
using namespace sofa::core::topology;
using namespace sofa::core::objectmodel;
using namespace sofa::component::topology;

/**
 * This class compute the intersection between two meshes
 */
template <class DataTypes>
class ComputeMeshIntersection : public sofa::core::DataEngine
{
public:
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef BaseMeshTopology::Triangle Triangle;
    typedef BaseMeshTopology::Edge Edge;
    typedef BaseMeshTopology::Quad Quad;
    typedef vector<BaseMeshTopology::Triangle> VecTriangles;
public:
    SOFA_CLASS(SOFA_TEMPLATE(ComputeMeshIntersection,DataTypes),sofa::core::DataEngine);

    ComputeMeshIntersection();

    ~ComputeMeshIntersection() {}

    void init();

    void update();

    Data<VecCoord> d_muscleLayerVertex, d_fatLayerVertex, d_intersectionVertex;
    Data<VecTriangles> d_muscleLayerTriangles, d_fatLayerTriangles, d_intersectionTriangles;
    Data< vector<Quad> > d_intersectionQuads;
    Data< vector<unsigned int> > d_index;

    MeshTopology topology;
    std::map<unsigned int, unsigned int> intersectionIndices;
    Data<bool> d_print_log;
    Data<double> d_epsilon;

    void computeIntersectionLayerVertex();
    void computeIntersectionLayerTriangles();
    bool isIntersectionLayerTriangle(const Triangle& ft, Triangle& fi);
    void closeMesh();

    void draw();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const ComputeMeshIntersection<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_PROGRESSIVESCALING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_ENGINE_API ComputeMeshIntersection<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_ENGINE_API ComputeMeshIntersection<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace pim

} // namespace plugins

#endif
