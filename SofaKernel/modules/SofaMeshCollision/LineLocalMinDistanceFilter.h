/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_LINELOCALMINDISTANCEFILTER_H
#define SOFA_COMPONENT_COLLISION_LINELOCALMINDISTANCEFILTER_H
#include "config.h"

#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/LocalMinDistanceFilter.h>
#include <SofaMeshCollision/PointLocalMinDistanceFilter.h>
#include <SofaBaseTopology/TopologyData.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace collision
{


/**
 * @brief LocalMinDistance cone information class for a Line collision primitive.
 */
class LineInfo : public InfoFilter
{
    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;

public:
    /**
     * @brief Default constructor.
     */
    LineInfo(LocalMinDistanceFilter *lmdFilters = nullptr);

    /**
     * @brief Default destructor.
     */
    ~LineInfo() override {}

    /**
     * @brief Returns the validity of a detected contact according to this LineInfo.
     */
    bool validate(const unsigned int edge_index, const defaulttype::Vector3& PQ) override;

    /**
     * @brief Output stream.
     */
    inline friend std::ostream& operator<< ( std::ostream& os, const LineInfo& /*ti*/ )
    {
        return os;
    }

    /**
     * @brief Input stream.
     */
    inline friend std::istream& operator>> ( std::istream& in, LineInfo& /*ti*/ )
    {
        return in;
    }


    /**
     * @brief Computes the region of interest cone of the Line primitive.
     */
    void buildFilter(unsigned int /*e*/) override;

protected:


    sofa::defaulttype::Vector3 m_nMean; ///<
    sofa::defaulttype::Vector3 m_triangleRight; ///<
    sofa::defaulttype::Vector3 m_triangleLeft; ///<
    sofa::defaulttype::Vector3 m_lineVector; ///<
    double	m_computedRightAngleCone; ///<
    double	m_computedLeftAngleCone; ///<
    bool	m_twoTrianglesAroundEdge; ///<
};


/**
 * @brief
 */
class SOFA_MESH_COLLISION_API LineLocalMinDistanceFilter : public LocalMinDistanceFilter
{
public:
    SOFA_CLASS(LineLocalMinDistanceFilter,sofa::component::collision::LocalMinDistanceFilter);

protected:
    LineLocalMinDistanceFilter();
    ~LineLocalMinDistanceFilter() override;

public:

    /**
     * @brief Scene graph initialization method.
     */
    void init() override;

    /**
     * @name These methods check the validity of a found intersection.
     * According to the local configuration around the found intersected primitive, we build a "Region Of Interest" geometric cone.
     * Pertinent intersections have to belong to this cone, others are not taking into account anymore.
     * If the filtration cone is unbuilt or invalid, these methods launch the build or update.
     */
    //@{


    /**
     * @brief Point Collision Primitive validation method.
     */
    bool validPoint(const int pointIndex, const defaulttype::Vector3 &PQ);



    /**
     * @brief Line Collision Primitive validation method.
     */
    bool validLine(const int /*lineIndex*/, const defaulttype::Vector3 &/*PQ*/);

    //@}

    /**
     * @brief New Points creations callback.
     */
    class PointInfoHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Point, helper::vector<PointInfo> >
    {
    public:
        PointInfoHandler(LineLocalMinDistanceFilter* _f, topology::PointData<helper::vector<PointInfo> >* _data) : topology::TopologyDataHandler<core::topology::BaseMeshTopology::Point, helper::vector<PointInfo> >(_data), f(_f) {}

        void applyCreateFunction(unsigned int pointIndex, PointInfo& m, const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);
    protected:
        LineLocalMinDistanceFilter* f;
    };

    /**
     * @brief New Edges creations callback.
     */
    class LineInfoHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, helper::vector<LineInfo> >
    {
    public:
        LineInfoHandler(LineLocalMinDistanceFilter* _f, topology::EdgeData<helper::vector<LineInfo> >* _data) : topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, helper::vector<LineInfo> >(_data), f(_f) {}

        void applyCreateFunction(unsigned int edgeIndex, LineInfo& m, const core::topology::BaseMeshTopology::Edge&, const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);
    protected:
        LineLocalMinDistanceFilter* f;
    };

private:
    topology::PointData< sofa::helper::vector<PointInfo> > m_pointInfo; ///< point filter data
    topology::EdgeData< sofa::helper::vector<LineInfo> > m_lineInfo; ///< line filter data

    PointInfoHandler* pointInfoHandler;
    LineInfoHandler* lineInfoHandler;

    core::topology::BaseMeshTopology *bmt;
};


} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_LINELOCALMINDISTANCEFILTER_H
