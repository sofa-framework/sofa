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
#include <sofa/component/engine/select/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/type/Vec.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::engine::select
{

/**
 * Select uniformly distributed points on a mesh based on Euclidean or Geodesic distance measure
 * The method uses farthest point sampling followed by Lloyd (k-means) relaxation
 * @author benjamin gilles
 */

template <class DataTypes>
class MeshSampler : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MeshSampler,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename Coord::value_type Real;

    typedef core::topology::BaseMeshTopology::PointID ID;
    typedef type::vector<ID> VI;
    typedef type::vector<VI> VVI;

    typedef type::vector<Real> VD;
    typedef type::vector<VD> VVD;

    typedef type::Vec<2,unsigned int> indicesType;

    typedef core::topology::BaseMeshTopology::SeqEdges SeqEdges;

public:

    MeshSampler();

    ~MeshSampler() override {}

    void reinit()    override { update();  }
    void init() override;
    void doUpdate() override;

    void draw(const core::visual::VisualParams* vparams) override;


    Data<unsigned int> number; ///< Sample number
    Data< VecCoord > position; ///< input positions
    Data< SeqEdges > f_edges;  ///< input edges for geodesic sampling
    Data< VecCoord > fixedPosition;  ///< User defined sample positions.
    Data<unsigned int> maxIter;     ///< Max number of LLoyd iterations
    Data< VI > outputIndices;       ///< selected point indices
    Data< VecCoord > outputPosition;       ///< selected point coordinates


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

#if !defined(SOFA_COMPONENT_ENGINE_MESHSAMPLER_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API MeshSampler<defaulttype::Vec3Types>;
 
#endif

} //namespace sofa::component::engine::select
