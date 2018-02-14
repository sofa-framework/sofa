/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_MESHSAMPLER_H
#define SOFA_COMPONENT_ENGINE_MESHSAMPLER_H
#include "config.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace engine
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
    typedef helper::vector<ID> VI;
    typedef helper::vector<VI> VVI;

    typedef helper::vector<Real> VD;
    typedef helper::vector<VD> VVD;

    typedef defaulttype::Vec<2,unsigned int> indicesType;

    typedef core::topology::BaseMeshTopology::SeqEdges SeqEdges;

public:

    MeshSampler();

    virtual ~MeshSampler() {}

    virtual void reinit()    override { update();  }
    virtual void init() override;
    void update() override;

    void draw(const core::visual::VisualParams* vparams) override;


    Data<unsigned int> number; ///< Sample number
    Data< VecCoord > position; ///< input positions
    Data< SeqEdges > f_edges;  ///< input edges for geodesic sampling
    Data< VecCoord > fixedPosition;  ///< User defined sample positions.
    Data<unsigned int> maxIter;     ///< Max number of LLoyd iterations
    Data< VI > outputIndices;       ///< selected point indices
    Data< VecCoord > outputPosition;       ///< selected point coordinates

    virtual std::string getTemplateName() const    override { return templateName(this);    }
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
extern template class SOFA_GENERAL_ENGINE_API MeshSampler<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API MeshSampler<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
