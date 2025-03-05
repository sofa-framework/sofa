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
#include <sofa/component/engine/analyze/config.h>



#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/type/SVector.h>


namespace sofa::component::engine::analyze
{

/**
 * This class groups points into overlapping clusters according to a user defined number of clusters and radius
 * It takes input positions (and optionally a meshtopology if geodesic distances are preferred)
 * output clusters can then be fed to
 *     - shape matching engine
 *     - blendSkinningMapping
 *
 */

template <class DataTypes>
class ClusteringEngine : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ClusteringEngine,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename Coord::value_type Real;

    typedef sofa::core::topology::BaseMeshTopology::PointID ID;
    typedef type::vector<ID> VI;
    typedef type::vector<VI> VVI;

    typedef type::vector<Real> VD;
    typedef type::vector<VD> VVD;

    typedef type::Vec<2,unsigned int> indicesType;

public:

    ClusteringEngine();

    ~ClusteringEngine() override {}

    void init() override;
    void doUpdate() override;

    void draw(const core::visual::VisualParams* vparams) override;

    Data<bool> d_useTopo; ///< Use available topology to compute neighborhood.
    //Data<unsigned int> maxIter;

    Data<Real> d_radius; ///< Neighborhood range.
    Data<Real> d_fixedRadius; ///< Neighborhood range (for non mechanical particles).
    Data<int> d_nbClusters; ///< Number of clusters (-1 means that all input points are selected).
    Data< VecCoord > d_fixedPosition; ///< Input positions of fixed (non mechanical) particles.
    Data< VecCoord > d_position; ///< Input rest positions.
    Data< VVI > d_cluster; ///< Computed clusters.

    sofa::core::objectmodel::DataFileName input_filename; ///< import precomputed clusters
    sofa::core::objectmodel::DataFileName output_filename; ///< export clusters

private:
    sofa::core::behavior::MechanicalState<DataTypes>* mstate;
    sofa::core::topology::BaseMeshTopology* topo;

    // recursively add to cluster[i] neighbors of lastNeighbors if dist<radius or if in voronoi(i)+one ring
    void AddNeighborhoodFromNeighborhood(VI& lastN, const unsigned int i, const VI& voronoi);

    // recursively add farthest point from already selected points
    void farthestPointSampling(VI& indices,VI& voronoi);

    // voronoi from a set of points -> returns voronoi and distances
    void Voronoi(const VI& indices , VD& distances, VI& voronoi);

    // dijkstra from a set of points -> returns voronoi and distances = Voronoi function with geodesic distances
    void dijkstra(const VI& indices , VD& distances, VI& voronoi);

    // relax farthest point sampling using LLoyd (k-means) algorithm
    void LLoyd();

    // IO
    bool load();
    std::string loadedFilename;
    bool save();
};

#if !defined(SOFA_COMPONENT_ENGINE_CLUSTERINGENGINE_CPP)
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API ClusteringEngine<defaulttype::Vec3Types>;
 
#endif

} //namespace sofa::component::engine::analyze
