/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <SofaMeshCollision/TriangleLocalMinDistanceFilter.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TopologyData.inl>

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

void TriangleInfo::buildFilter(unsigned int tri_index)
{


    sofa::core::topology::BaseMeshTopology* bmt = this->base_mesh_topology;
    const Triangle &t =  bmt->getTriangle(tri_index);

    const sofa::defaulttype::Vector3 &pt1 = (*this->position_filtering)[t[0]];
    const sofa::defaulttype::Vector3 &pt2 = (*this->position_filtering)[t[1]];
    const sofa::defaulttype::Vector3 &pt3 = (*this->position_filtering)[t[2]];

    m_normal = cross(pt2-pt1, pt3-pt1);


    setValid();
}



bool TriangleInfo::validate(const unsigned int tri_index, const defaulttype::Vector3 &PQ)
{
    if (isValid())
    {
        return ( (m_normal * PQ) >= 0.0 );
    }
    else
    {
        buildFilter(tri_index);
        return validate(tri_index, PQ);
    }
}


TriangleLocalMinDistanceFilter::TriangleLocalMinDistanceFilter()
    : m_pointInfo(initData(&m_pointInfo, "pointInfo", "point filter data"))
    , m_lineInfo(initData(&m_lineInfo, "lineInfo", "line filter data"))
    , m_triangleInfo(initData(&m_triangleInfo, "triangleInfo", "triangle filter data"))
    , pointInfoHandler(NULL)
    , lineInfoHandler(NULL)
    , triangleInfoHandler(NULL)
    , bmt(NULL)
{
}

TriangleLocalMinDistanceFilter::~TriangleLocalMinDistanceFilter()
{
    if (pointInfoHandler) delete pointInfoHandler;
    if (lineInfoHandler) delete lineInfoHandler;
    if (triangleInfoHandler) delete triangleInfoHandler;
}

void TriangleLocalMinDistanceFilter::init()
{
    bmt = getContext()->getMeshTopology();
    std::cout<<"Mesh Topology found :"<<bmt->getName()<<std::endl;
    component::container::MechanicalObject<sofa::defaulttype::Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<Vec3Types>*>(getContext()->getMechanicalState());


    if(mstateVec3d == NULL)
    {
        serr<<"WARNING: init failed for TriangleLocalMinDistanceFilter no mstateVec3d found"<<sendl;
    }

    if (bmt != 0)
    {

        pointInfoHandler = new PointInfoHandler(this,&m_pointInfo);
        m_pointInfo.createTopologicalEngine(bmt, pointInfoHandler);
        m_pointInfo.registerTopologicalData();
        m_pointInfo.createTopologicalEngine(bmt);

        helper::vector< PointInfo >& pInfo = *(m_pointInfo.beginEdit());
        pInfo.resize(bmt->getNbPoints());
        int i;
        for (i=0; i<bmt->getNbPoints(); i++)
        {
            pInfo[i].setLMDFilters(this);
            pInfo[i].setBaseMeshTopology(bmt);
            pInfo[i].setPositionFiltering(&mstateVec3d->read(core::ConstVecCoordId::position())->getValue());
        }
        m_pointInfo.endEdit();

        lineInfoHandler = new LineInfoHandler(this,&m_lineInfo);
        m_lineInfo.createTopologicalEngine(bmt, lineInfoHandler);
        m_lineInfo.registerTopologicalData();

        helper::vector< LineInfo >& lInfo = *(m_lineInfo.beginEdit());
        lInfo.resize(bmt->getNbEdges());
        for (i=0; i<bmt->getNbEdges(); i++)
        {
            lInfo[i].setLMDFilters(this);
            lInfo[i].setBaseMeshTopology(bmt);
            lInfo[i].setPositionFiltering(&mstateVec3d->read(core::ConstVecCoordId::position())->getValue());
        }
        m_lineInfo.endEdit();

        triangleInfoHandler = new TriangleInfoHandler(this,&m_triangleInfo);
        m_triangleInfo.createTopologicalEngine(bmt, triangleInfoHandler);
        m_triangleInfo.registerTopologicalData();

        helper::vector< TriangleInfo >& tInfo = *(m_triangleInfo.beginEdit());
        tInfo.resize(bmt->getNbTriangles());
        for (i=0; i<bmt->getNbTriangles(); i++)
        {
            tInfo[i].setLMDFilters(this);
            tInfo[i].setBaseMeshTopology(bmt);
            tInfo[i].setPositionFiltering(&mstateVec3d->read(core::ConstVecCoordId::position())->getValue());
        }
        m_triangleInfo.endEdit();
        std::cout<<"create m_pointInfo, m_lineInfo, m_triangleInfo" <<std::endl;
    }

    if(this->isRigid())
    {
        std::cout<<"++++++ Is rigid Found in init "<<std::endl;
        // Precomputation of the filters in the rigid case
        //triangles:
        helper::vector< TriangleInfo >& tInfo = *(m_triangleInfo.beginEdit());
        for(unsigned int t=0; t<tInfo.size(); t++)
        {
            tInfo[t].buildFilter(t);

        }
        m_triangleInfo.endEdit();

        //lines:
        helper::vector< LineInfo >& lInfo = *(m_lineInfo.beginEdit());
        for(unsigned int l=0; l<lInfo.size(); l++)
        {
            lInfo[l].buildFilter(l);

        }
        m_lineInfo.endEdit();

        //points:
        helper::vector< PointInfo >& pInfo = *(m_pointInfo.beginEdit());
        for(unsigned int p=0; p<pInfo.size(); p++)
        {
            pInfo[p].buildFilter(p);

        }
        m_pointInfo.endEdit();

    }

}



void TriangleLocalMinDistanceFilter::handleTopologyChange()
{
    if(this->isRigid())
    {
        serr<<"WARNING: filters optimization needed for topological change on rigid collision model"<<sendl;
        this->invalidate(); // all the filters will be recomputed, not only those involved in the topological change
    }
}



void TriangleLocalMinDistanceFilter::PointInfoHandler::applyCreateFunction(unsigned int /*pointIndex*/, PointInfo &pInfo, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const TriangleLocalMinDistanceFilter *tLMDFilter = this->f;
    pInfo.setLMDFilters(tLMDFilter);
    sofa::core::topology::BaseMeshTopology * bmt = tLMDFilter->bmt;
    pInfo.setBaseMeshTopology(bmt);
    /////// TODO : template de la classe
    component::container::MechanicalObject<Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<Vec3Types>*>(tLMDFilter->getContext()->getMechanicalState());
    if(tLMDFilter->isRigid())
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            pInfo.setPositionFiltering(&(mstateVec3d->read(core::ConstVecCoordId::restPosition())->getValue()));
        }

    }
    else
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            pInfo.setPositionFiltering(&mstateVec3d->read(core::ConstVecCoordId::position())->getValue());
        }
    }

}



void TriangleLocalMinDistanceFilter::LineInfoHandler::applyCreateFunction(unsigned int /*edgeIndex*/, LineInfo &lInfo, const core::topology::BaseMeshTopology::Edge&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const TriangleLocalMinDistanceFilter *tLMDFilter = this->f;
    lInfo.setLMDFilters(tLMDFilter);
    sofa::core::topology::BaseMeshTopology * bmt = tLMDFilter->bmt; // (sofa::core::topology::BaseMeshTopology *)tLMDFilter->getContext()->getTopology();
    lInfo.setBaseMeshTopology(bmt);
    /////// TODO : template de la classe
    component::container::MechanicalObject<Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<Vec3Types>*>(tLMDFilter->getContext()->getMechanicalState());
    if(tLMDFilter->isRigid())
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            lInfo.setPositionFiltering(&(mstateVec3d->read(core::ConstVecCoordId::restPosition())->getValue()));
        }

    }
    else
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            lInfo.setPositionFiltering(&mstateVec3d->read(core::ConstVecCoordId::position())->getValue());
        }
    }
}



void TriangleLocalMinDistanceFilter::TriangleInfoHandler::applyCreateFunction(unsigned int /*edgeIndex*/, TriangleInfo &tInfo, const core::topology::BaseMeshTopology::Triangle&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const TriangleLocalMinDistanceFilter *tLMDFilter = this->f;
    tInfo.setLMDFilters(tLMDFilter);
    sofa::core::topology::BaseMeshTopology * bmt = tLMDFilter->bmt; // (sofa::core::topology::BaseMeshTopology *)tLMDFilter->getContext()->getTopology();
    tInfo.setBaseMeshTopology(bmt);
    /////// TODO : template de la classe
    component::container::MechanicalObject<Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<Vec3Types>*>(tLMDFilter->getContext()->getMechanicalState());
    if(tLMDFilter->isRigid())
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            tInfo.setPositionFiltering(&(mstateVec3d->read(core::ConstVecCoordId::restPosition())->getValue()));
        }

    }
    else
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            tInfo.setPositionFiltering(&mstateVec3d->read(core::ConstVecCoordId::position())->getValue());
        }
    }
}


bool TriangleLocalMinDistanceFilter::validPoint(const int pointIndex, const defaulttype::Vector3 &PQ)
{
    PointInfo & Pi = m_pointInfo[pointIndex];

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


bool TriangleLocalMinDistanceFilter::validLine(const int lineIndex, const defaulttype::Vector3 &PQ)
{
    LineInfo &Li = m_lineInfo[lineIndex];  // filter is precomputed

    if(this->isRigid())
    {
        defaulttype::Vector3 PQtest;
        PQtest = pos->getOrientation().inverseRotate(PQ);
        return Li.validate(lineIndex,PQtest);
    }

    return Li.validate(lineIndex, PQ);
}


bool TriangleLocalMinDistanceFilter::validTriangle(const int triangleIndex, const defaulttype::Vector3 &PQ)
{
    TriangleInfo &Ti = m_triangleInfo[triangleIndex];

    if(this->isRigid())
    {
        defaulttype::Vector3 PQtest;
        PQtest = pos->getOrientation().inverseRotate(PQ);
        return Ti.validate(triangleIndex,PQtest);
    }

    return Ti.validate(triangleIndex,PQ);
}



SOFA_DECL_CLASS(TriangleLocalMinDistanceFilter)

int TriangleLocalMinDistanceFilterClass = core::RegisterObject("This class manages Triangle collision models cones filters computations and updates.")
        .add< TriangleLocalMinDistanceFilter >()
        ;

} // namespace collision

} // namespace component

} // namespace sofa
