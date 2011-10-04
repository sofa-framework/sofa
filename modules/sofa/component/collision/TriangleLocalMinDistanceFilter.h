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
#ifndef SOFA_COMPONENT_COLLISION_TRIANGLELOCALMINDISTANCEFILTER_H
#define SOFA_COMPONENT_COLLISION_TRIANGLELOCALMINDISTANCEFILTER_H

#include <sofa/component/collision/LocalMinDistanceFilter.h>
#include <sofa/component/collision/LineLocalMinDistanceFilter.h>
#include <sofa/component/collision/PointLocalMinDistanceFilter.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/topology/EdgeData.h>
#include <sofa/component/topology/PointData.h>
#include <sofa/component/topology/TriangleData.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace component
{

namespace collision
{
using sofa::helper::AdvancedTimer;

/**
 * @brief LocalMinDistance cone information class for a Triangle collision primitive.
 */
class TriangleInfo : public InfoFilter //< Triangle >
{
    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
public:
    /**
     * @brief Default constructor.
     */
    TriangleInfo(LocalMinDistanceFilter *lmdFilters)
        : InfoFilter(lmdFilters)
    {

    }

    /**
     * @brief Empty constructor. Required by TriangleData<>.
     */
    TriangleInfo()
        : InfoFilter(NULL)
    {

    }

    /**
     * @brief Default destructor.
     */
    virtual ~TriangleInfo() {};

    /**
     * @brief Returns the validity of a detected contact according to this TriangleInfo.
     */
    virtual bool validate(const unsigned int /*p*/, const defaulttype::Vector3 & /*PQ*/);
    /**
     * @brief Output stream.
     */
    inline friend std::ostream& operator<< ( std::ostream& os, const TriangleInfo& /*ti*/ )
    {
        return os;
    }

    /**
     * @brief Input stream.
     */
    inline friend std::istream& operator>> ( std::istream& in, TriangleInfo& /*ti*/ )
    {
        return in;
    }

    /**
     * @brief Computes the region of interest cone of the Triangle primitive.
     */
    //virtual void buildFilter(const Triangle & /*t*/);
    virtual void buildFilter(unsigned int /*t*/);

protected:


    Vector3 m_normal; ///< Stored normal of the triangle.
};



/**
 * @brief
 */
class SOFA_COMPONENT_COLLISION_API TriangleLocalMinDistanceFilter : public LocalMinDistanceFilter
{
public:
    SOFA_CLASS(TriangleLocalMinDistanceFilter, LocalMinDistanceFilter);

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
    bool validPoint(const int pointIndex, const defaulttype::Vector3 &PQ)
    {
        // AdvancedTimer::StepVar("Filters");

        PointInfo & Pi = m_pointInfo[pointIndex];
        if(&Pi==NULL)
        {
            serr<<"Pi == NULL"<<sendl;
            return true;
        }

        if(this->isRigid())
        {
            // filter is precomputed in the rest position
            defaulttype::Vector3 PQtest;
            PQtest = pos->getOrientation().inverseRotate(PQ);
            return Pi.validate(pointIndex,PQtest);
        }
        //else

        return Pi.validate(pointIndex,PQ);
    }

    /**
     * @brief Line Collision Primitive validation method.
     */
    bool validLine(const int lineIndex, const defaulttype::Vector3 &PQ)
    {
        //AdvancedTimer::StepVar("Filters");

        LineInfo &Li = m_lineInfo[lineIndex];  // filter is precomputed
        if(&Li==NULL)
        {
            serr<<"Li == NULL"<<sendl;
            return true;
        }

        if(this->isRigid())
        {
            defaulttype::Vector3 PQtest;
            PQtest = pos->getOrientation().inverseRotate(PQ);
            return Li.validate(lineIndex,PQtest);
        }

        //std::cout<<"validLine "<<lineIndex<<" is called with PQ="<<PQ<<std::endl;
        return Li.validate(lineIndex, PQ);
    }

    /**
     * @brief Triangle Collision Primitive validation method.
     */
    bool validTriangle(const int triangleIndex, const defaulttype::Vector3 &PQ)
    {
        //AdvancedTimer::StepVar("Filters");
        //std::cout<<"validTriangle "<<triangleIndex<<" is called with PQ="<<PQ<<std::endl;
        TriangleInfo &Ti = m_triangleInfo[triangleIndex];

        if(&Ti==NULL)
        {
            serr<<"Ti == NULL"<<sendl;
            return true;
        }

        if(this->isRigid())
        {
            defaulttype::Vector3 PQtest;
            PQtest = pos->getOrientation().inverseRotate(PQ);
            return Ti.validate(triangleIndex,PQtest);
        }


        return Ti.validate(triangleIndex,PQ);
    }

    //@}

    /**
     * @brief New Points creations callback.
     */
    static void LMDFilterPointCreationFunction(unsigned int, void*, PointInfo &, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&);

    /**
     * @brief New Edges creations callback.
     */
    static void LMDFilterLineCreationFunction(unsigned int , void*, LineInfo &, const topology::Edge&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&);

    /**
     * @brief New Triangles creations callback.
     */
    static void LMDFilterTriangleCreationFunction(unsigned int , void*, TriangleInfo &, const topology::Triangle&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&);

private:
    topology::PointData< sofa::helper::vector<PointInfo> > m_pointInfo;
    topology::EdgeData< sofa::helper::vector<LineInfo> > m_lineInfo;
    topology::TriangleData< sofa::helper::vector<TriangleInfo> > m_triangleInfo;
};


} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_TRIANGLELOCALMINDISTANCEFILTER_H
