/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ENGINE_MESHSAMPLER_H
#define SOFA_COMPONENT_ENGINE_MESHSAMPLER_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.inl>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/SVector.h>

#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace core::behavior;
using namespace core::topology;
using namespace core::objectmodel;

/**
 * Select uniformly distributed points on a mesh based on Euclidean or Geodesic distance measure
 * The method uses farthest point sampling followed by Lloyd (k-means) relaxation
 */

template <class DataTypes>
class MeshSampler : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MeshSampler,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename Coord::value_type Real;

    typedef BaseMeshTopology::PointID ID;
    typedef helper::vector<ID> VI;
    typedef helper::vector<VI> VVI;

    typedef helper::vector<Real> VD;
    typedef helper::vector<VD> VVD;

    typedef defaulttype::Vec<2,unsigned int> indicesType;

    typedef BaseMeshTopology::SeqEdges SeqEdges;

public:

    MeshSampler();

    virtual ~MeshSampler() {}

    void init();
    void update();

    void draw(const core::visual::VisualParams* vparams);


    Data<unsigned int> number;
    Data< VecCoord > position; ///< input positions
    Data< SeqEdges > f_edges;  ///< input edges for geodesic sampling
    Data< VecCoord > fixedPosition;  ///< User defined sample positions.
    Data<unsigned int> maxIter;     ///< Max number of LLoyd iterations
    Data< VI > outputIndices;       ///< selected point indices
    Data< VecCoord > outputPosition;       ///< selected point coordinates

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const MeshSampler<DataTypes>* = NULL) {   return DataTypes::Name(); }

private:
    // recursively add farthest point from already selected points
    void farthestPointSampling(VD& distances,VI& voronoi,const VVI& ngb);

    // returns index and distances from positions to their closest sample
    void computeDistances(VD& distances, VI& voronoi,const VVI& ngb);

    // relax farthest point sampling using LLoyd (k-means) algorithm
    bool LLoyd(VD& distances,VI& voronoi,const VVI& ngb);

    // one ring neighbors from edges
    void computeNeighbors(VVI& ngb);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_MESHSAMPLER_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_ENGINE_API MeshSampler<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_ENGINE_API MeshSampler<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
