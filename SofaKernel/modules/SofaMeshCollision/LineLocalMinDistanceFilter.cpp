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

#include <SofaMeshCollision/LineLocalMinDistanceFilter.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/TopologyData.inl>

#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{


void LineInfo::buildFilter(unsigned int edge_index)
{
    using sofa::helper::vector;
    using sofa::core::topology::BaseMeshTopology;

    bool debug=false;

    if ((int)edge_index==-1)
        debug=true;

    BaseMeshTopology* bmt = this->base_mesh_topology;

    const Edge &e =  bmt->getEdge(edge_index);

    const sofa::defaulttype::Vector3 &pt1 = (*this->position_filtering)[e[0]];
    const sofa::defaulttype::Vector3 &pt2 = (*this->position_filtering)[e[1]];

    if (debug)
        std::cout<<"pt1: "<<pt1<<"  - pt2: "<<pt2;

    m_lineVector = pt2 - pt1;
    m_lineVector.normalize();

    const sofa::helper::vector<unsigned int>& trianglesAroundEdge = bmt->getTrianglesAroundEdge(edge_index);

    if (debug)
        std::cout<<"trianglesAroundEdge: "<<trianglesAroundEdge<<"  -";

    // filter if there are two triangles around the edge
    if (trianglesAroundEdge.size() == 1)
    {
        std::cout<<"TODO : validity for segment on a single triangle"<<std::endl;
    }

    // filter if there are two triangles around the edge
    if (trianglesAroundEdge.size() != 2)
    {
        m_twoTrianglesAroundEdge = false;
        return;
    }

    const sofa::helper::vector<sofa::defaulttype::Vector3>& x = *this->position_filtering;



    // which triangle is left ?
    const Triangle& triangle0 = bmt->getTriangle(trianglesAroundEdge[0]);
    bool triangle0_is_left=false;
    if ( (e[0]==triangle0[0]&&e[1]==triangle0[1]) || (e[0]==triangle0[1]&&e[1]==triangle0[2]) || (e[0]==triangle0[2]&&e[1]==triangle0[0]) )
        triangle0_is_left=true;

    // compute the normal of the triangle situated on the right
    const BaseMeshTopology::Triangle& triangleRight = triangle0_is_left ? bmt->getTriangle(trianglesAroundEdge[1]): bmt->getTriangle(trianglesAroundEdge[0]);
    sofa::defaulttype::Vector3 n1 = cross(x[triangleRight[1]] - x[triangleRight[0]], x[triangleRight[2]] - x[triangleRight[0]]);
    n1.normalize();
    m_nMean = n1;
    m_triangleRight = cross(n1, m_lineVector);
    m_triangleRight.normalize(); // necessary ?

    // compute the normal of the triangle situated on the left
    const BaseMeshTopology::Triangle& triangleLeft = triangle0_is_left ? bmt->getTriangle(trianglesAroundEdge[0]): bmt->getTriangle(trianglesAroundEdge[1]);
    sofa::defaulttype::Vector3 n2 = cross(x[triangleLeft[1]] - x[triangleLeft[0]], x[triangleLeft[2]] - x[triangleLeft[0]]);
    n2.normalize();
    m_nMean += n2;
    m_triangleLeft = cross(m_lineVector, n2);
    m_triangleLeft.normalize(); // necessary ?

    m_nMean.normalize();

    // compute the angle for the cone to filter contacts using the normal of the triangle situated on the right
    m_computedRightAngleCone = (m_nMean * m_triangleRight) * m_lmdFilters->getConeExtension();
    if(debug)
        std::cout<<"m_nMean: "<<m_nMean<<" - m_triangleRight:"<<m_triangleRight<<" - m_triangleLeft:"<<m_triangleLeft<<std::endl;
    if (m_computedRightAngleCone < 0)
    {
        m_computedRightAngleCone = 0.0;
    }
    m_computedRightAngleCone += m_lmdFilters->getConeMinAngle();
    if( debug)
        std::cout<<"m_computedRightAngleCone :"<<m_computedRightAngleCone<<std::endl;

    // compute the angle for the cone to filter contacts using the normal of the triangle situated on the left
    m_computedLeftAngleCone = (m_nMean * m_triangleLeft) * m_lmdFilters->getConeExtension();
    if (m_computedLeftAngleCone < 0)
    {
        m_computedLeftAngleCone = 0.0;
    }
    m_computedLeftAngleCone += m_lmdFilters->getConeMinAngle();
    if( debug)
        std::cout<<"m_computedLeftAngleCone :"<<m_computedRightAngleCone<<std::endl;


    setValid();
}




//bool LineInfo::validate(const unsigned int edge_index, const defaulttype::Vector3 &PQ)

bool LineInfo::validate(const unsigned int edge_index, const defaulttype::Vector3& PQ)
{
    bool debug=false;

    if ((int)edge_index==-1)
        debug=true;



    if (isValid())
    {
        if (debug)
            std::cout<<"Line "<<edge_index<<" is valid"<<std::endl;
        if (m_twoTrianglesAroundEdge)
        {
            if (debug)
            {
                std::cout<<"m_triangleRight :"<<m_triangleRight<<"  - m_triangleLeft"<<m_triangleLeft<<std::endl;
                std::cout<<"m_twoTrianglesAroundEdge ok tests: "<< (m_nMean * PQ)<<"<0 ?  - "<<m_triangleRight * PQ <<" < "<<-m_computedRightAngleCone * PQ.norm()<<" ?  - " <<m_triangleLeft * PQ <<" < "<<-m_computedLeftAngleCone * PQ.norm()<<" ?"<<std::endl;

            }
            if ((m_nMean * PQ) < 0)
                return false;

            if (m_triangleRight * PQ < -m_computedRightAngleCone * PQ.norm())
                return false;

            if (m_triangleLeft * PQ < -m_computedLeftAngleCone * PQ.norm())
                return false;
        }
        else
        {
            sofa::defaulttype::Vector3 PQnormalized = PQ;
            PQnormalized.normalize();

            if (fabs(dot(m_lineVector, PQnormalized)) > m_lmdFilters->getConeMinAngle() + 0.001)		// dot(AB,n1) should be equal to 0
            {
                // means that proximity was detected with a null determinant
                // in function computeIntersection
                return false;
            }
        }

        return true;
    }
    else
    {
        if (debug)
            std::cout<<"Line "<<edge_index<<" is no valid ------------ build"<<std::endl;
        buildFilter(edge_index);
        return validate(edge_index, PQ);
    }
}

LineLocalMinDistanceFilter::LineLocalMinDistanceFilter()
    : m_pointInfo(initData(&m_pointInfo, "pointInfo", "point filter data"))
    , m_lineInfo(initData(&m_lineInfo, "lineInfo", "line filter data"))
    , pointInfoHandler(NULL)
    , lineInfoHandler(NULL)
    , bmt(NULL)
{
}

LineLocalMinDistanceFilter::~LineLocalMinDistanceFilter()
{
    if (pointInfoHandler) delete pointInfoHandler;
    if (lineInfoHandler) delete lineInfoHandler;
}

void LineLocalMinDistanceFilter::init()
{
    this->bmt = getContext()->getMeshTopology();

    if (bmt != 0)
    {
        helper::vector< PointInfo >& pInfo = *(m_pointInfo.beginEdit());
        pInfo.resize(bmt->getNbPoints());
        m_pointInfo.endEdit();

        pointInfoHandler = new PointInfoHandler(this,&m_pointInfo);
        m_pointInfo.createTopologicalEngine(bmt, pointInfoHandler);
        m_pointInfo.registerTopologicalData();

        helper::vector< LineInfo >& lInfo = *(m_lineInfo.beginEdit());
        lInfo.resize(bmt->getNbEdges());
        m_lineInfo.endEdit();

        lineInfoHandler = new LineInfoHandler(this,&m_lineInfo);
        m_lineInfo.createTopologicalEngine(bmt, lineInfoHandler);
        m_lineInfo.registerTopologicalData();
    }
}



void LineLocalMinDistanceFilter::PointInfoHandler::applyCreateFunction(unsigned int /*pointIndex*/, PointInfo &pInfo, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const LineLocalMinDistanceFilter *lLMDFilter = this->f;
    pInfo.setLMDFilters(lLMDFilter);

    sofa::core::topology::BaseMeshTopology * bmt = lLMDFilter->bmt; //getContext()->getTopology();
    pInfo.setBaseMeshTopology(bmt);
    /////// TODO : template de la classe
    component::container::MechanicalObject<sofa::defaulttype::Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<sofa::defaulttype::Vec3Types>*>(lLMDFilter->getContext()->getMechanicalState());
    if(mstateVec3d != NULL)
    {
        pInfo.setPositionFiltering(&mstateVec3d->read(core::ConstVecCoordId::position())->getValue());
    }

    //component::container::MechanicalObject<Vec3fTypes>*  mstateVec3f= dynamic_cast<component::container::MechanicalObject<Vec3fTypes>*>(context->getMechanicalState())
    //if(mstateVec3f != NULL)
    //{
    //	lInfo.setPositionFiltering(mstateVec3f->read(sofa::core::ConstVecCoordId::position())->getValue());
    //}
}



void LineLocalMinDistanceFilter::LineInfoHandler::applyCreateFunction(unsigned int /*edgeIndex*/, LineInfo &lInfo, const core::topology::BaseMeshTopology::Edge&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const LineLocalMinDistanceFilter *lLMDFilter = this->f;
    lInfo.setLMDFilters(lLMDFilter);
    //
    sofa::core::topology::BaseMeshTopology * bmt = lLMDFilter->bmt; //getContext()->getTopology();
    lInfo.setBaseMeshTopology(bmt);


    /////// TODO : template de la classe
    component::container::MechanicalObject<sofa::defaulttype::Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<sofa::defaulttype::Vec3Types>*>(lLMDFilter->getContext()->getMechanicalState());
    if(mstateVec3d != NULL)
    {
        lInfo.setPositionFiltering(&mstateVec3d->read(core::ConstVecCoordId::position())->getValue());
    }

    //component::container::MechanicalObject<Vec3fTypes>*  mstateVec3f= dynamic_cast<component::container::MechanicalObject<Vec3fTypes>*>(context->getMechanicalState())
    //if(mstateVec3f != NULL)
    //{
    //	lInfo.setPositionFiltering(mstateVec3f->read(sofa::core::ConstVecCoordId::position())->getValue());
    //}

}

bool LineLocalMinDistanceFilter::validPoint(const int pointIndex, const defaulttype::Vector3 &PQ)
{

    PointInfo & Pi = m_pointInfo[pointIndex];
//    if(&Pi==NULL)
//    {
//        serr<<"Pi == NULL"<<sendl;
//        return true;
//    }

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

bool LineLocalMinDistanceFilter::validLine(const int /*lineIndex*/, const defaulttype::Vector3 &/*PQ*/)
{
    //const Edge& bmt->getEdge(lineIndex);
    // m_lineInfo[edgeIndex].validate(lineIndex, PQ);
    return true;
}


SOFA_DECL_CLASS(LineLocalMinDistanceFilter)

int LineLocalMinDistanceFilterClass = core::RegisterObject("This class manages Line collision models cones filters computations and updates.")
        .add< LineLocalMinDistanceFilter >()
        ;

} // namespace collision

} // namespace component

} // namespace sofa
