/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_COLLISION_POINTLOCALMINDISTANCEFILTER_H
#define SOFA_COMPONENT_COLLISION_POINTLOCALMINDISTANCEFILTER_H

#include <SofaMeshCollision/LocalMinDistanceFilter.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseTopology/TopologyData.h>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa
{

namespace component
{

namespace collision
{


/**
 * @brief LocalMinDistance cone information class for a Point collision primitive.
 */
class PointInfo : public InfoFilter //< Point >
{
public:
    typedef std::vector< std::pair< sofa::defaulttype::Vector3, double > > TDataContainer;

    /**
     * @brief Default constructor.
     */
    PointInfo(LocalMinDistanceFilter *lmdFilters)
        : InfoFilter(lmdFilters)
    {

    }

    /**
     * @brief Empty constructor. Required by PointData<>.
     */
    PointInfo()
        : InfoFilter(NULL)
    {

    }

    /**
     * @brief Default destructor.
     */
    virtual ~PointInfo() {};

    /**
     * @brief Returns the validity of a detected contact according to this PointInfo.
     */
    //virtual bool validate(const Point & /*p*/, const defaulttype::Vector3 & /*PQ*/);
    virtual bool validate(const unsigned int /*p*/, const defaulttype::Vector3 & /*PQ*/);
    /**
     * @brief Output stream.
     */
    inline friend std::ostream& operator<< ( std::ostream& os, const PointInfo& /*ti*/ )
    {
        return os;
    }

    /**
     * @brief Input stream.
     */
    inline friend std::istream& operator>> ( std::istream& in, PointInfo& /*ti*/ )
    {
        return in;
    }

    /**
     * @brief Computes the region of interest cone of the Point primitive.
     */
    //virtual void buildFilter(const Point & /*p*/);
    virtual void buildFilter(unsigned int /*p*/);

protected:


    bool m_noLineModel; ///< Flag indicating if the Point CollisionModel is not associated to a Line CollisionModel.
    TDataContainer m_computedData; ///< Cone stored data.
};


/**
 * @brief
 */
class SOFA_MESH_COLLISION_API PointLocalMinDistanceFilter : public LocalMinDistanceFilter
{
public:
    SOFA_CLASS(PointLocalMinDistanceFilter,LocalMinDistanceFilter);

protected:
    PointLocalMinDistanceFilter();
    virtual ~PointLocalMinDistanceFilter();
public:

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
    bool validPoint(int /*pointIndex*/, const defaulttype::Vector3 &/*PQ*/)
    {
        return true;
    }

    //@}

    /**
     * @brief New Points creations handler.
     */
    class PointInfoHandler : public topology::TopologyDataHandler<topology::Point, helper::vector<PointInfo> >
    {
    public:
        PointInfoHandler(PointLocalMinDistanceFilter* _f, topology::PointData<helper::vector<PointInfo> >* _data) : topology::TopologyDataHandler<topology::Point, helper::vector<PointInfo> >(_data), f(_f) {}

        void applyCreateFunction(unsigned int pointIndex, PointInfo& m, const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);
    protected:
        PointLocalMinDistanceFilter* f;
    };

private:
    topology::PointData< sofa::helper::vector<PointInfo> > m_pointInfo;
    PointInfoHandler* pointInfoHandler;
    core::topology::BaseMeshTopology *bmt;
};


} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_POINTLOCALMINDISTANCEFILTER_H
