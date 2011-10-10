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
#ifndef SOFA_COMPONENT_COLLISION_LINELOCALMINDISTANCEFILTER_H
#define SOFA_COMPONENT_COLLISION_LINELOCALMINDISTANCEFILTER_H

#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/LocalMinDistanceFilter.h>
#include <sofa/component/collision/PointLocalMinDistanceFilter.h>
#include <sofa/component/topology/EdgeData.h>
#include <sofa/component/topology/PointData.h>

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
class LineInfo : public InfoFilter //< topology::Edge >
{
    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;

public:
    /**
     * @brief Empty constructor. Required by EdgeData<>.
     */
    LineInfo()
        : InfoFilter(NULL)
    {
        todo=false;
    }

    /**
     * @brief Default constructor.
     */
    LineInfo(LocalMinDistanceFilter *lmdFilters)
        : InfoFilter(lmdFilters)
    {
        todo=false;
    }

    /**
     * @brief Default destructor.
     */
    virtual ~LineInfo() {};

    /**
     * @brief Returns the validity of a detected contact according to this LineInfo.
     */
    virtual bool validate(const unsigned int edge_index, const defaulttype::Vector3& PQ);

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
    virtual void buildFilter(unsigned int /*e*/);

protected:


    Vector3 m_nMean; ///<
    Vector3 m_triangleRight; ///<
    Vector3 m_triangleLeft; ///<
    Vector3 m_lineVector; ///<
    double	m_computedRightAngleCone; ///<
    double	m_computedLeftAngleCone; ///<
    bool	m_twoTrianglesAroundEdge; ///<
    bool todo;
};


/**
 * @brief
 */
class SOFA_BASE_COLLISION_API LineLocalMinDistanceFilter : public LocalMinDistanceFilter
{
public:
    SOFA_CLASS(LineLocalMinDistanceFilter,sofa::component::collision::LocalMinDistanceFilter);

    /**
     * @brief Scene graph initialization method.
     */
    void init();

    /**
     * @brief Handle topological changes.
     */
    void handleTopologyChange();

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
    static void LMDFilterPointCreationFunction(unsigned int, void*, PointInfo &, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&);

    /**
     * @brief New Edges creations callback.
     */
    static void LMDFilterLineCreationFunction(unsigned int , void*, LineInfo &, const topology::Edge&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&);

private:
    topology::PointData< sofa::helper::vector<PointInfo> > m_pointInfo;
    topology::EdgeData< sofa::helper::vector<LineInfo> > m_lineInfo;

    core::topology::BaseMeshTopology *bmt;


};


} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_LINELOCALMINDISTANCEFILTER_H
