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

#include <sofa/component/collision/TriangleLocalMinDistanceFilter.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/TopologyData.inl>

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{

void TriangleInfo::buildFilter(unsigned int tri_index)
{


    sofa::core::topology::BaseMeshTopology* bmt = this->base_mesh_topology;
    const Triangle &t =  bmt->getTriangle(tri_index);

    const Vector3 &pt1 = (*this->position_filtering)[t[0]];
    const Vector3 &pt2 = (*this->position_filtering)[t[1]];
    const Vector3 &pt3 = (*this->position_filtering)[t[2]];

    m_normal = cross(pt2-pt1, pt3-pt1);


    setValid();
}



bool TriangleInfo::validate(const unsigned int tri_index, const defaulttype::Vector3 &PQ)
{
    //std::cout<<"TriangleInfo::validate on tri "<<tri_index<<"is called"<<std::endl;
    if (isValid())
    {
        //std::cout<<" is Valid !"<<std::endl;
        return ( (m_normal * PQ) >= 0.0 );
    }
    else
    {
        //std::cout<<" not valid => build ------------------------ for triangle "<< tri_index <<std::endl;
        buildFilter(tri_index);
        return validate(tri_index, PQ);
    }
}



void TriangleLocalMinDistanceFilter::init()
{
    core::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();
    std::cout<<"Mesh Topology found :"<<bmt->getName()<<std::endl;
    component::container::MechanicalObject<Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<Vec3Types>*>(getContext()->getMechanicalState());


    if(mstateVec3d == NULL)
    {
        serr<<"WARNING: init failed for TriangleLocalMinDistanceFilter no mstateVec3d found"<<sendl;
    }

    if (bmt != 0)
    {

        m_pointInfo.createTopologicalEngine(bmt);
#ifdef TODOTOPO
        m_pointInfo.setCreateFunction(LMDFilterPointCreationFunction);
        m_pointInfo.setCreateParameter((void *) this);
        m_pointInfo.setDestroyParameter( (void *) this );
#endif
        m_pointInfo.registerTopologicalData();

        helper::vector< PointInfo >& pInfo = *(m_pointInfo.beginEdit());
        pInfo.resize(bmt->getNbPoints());
        int i;
        for (i=0; i<bmt->getNbPoints(); i++)
        {
            pInfo[i].setLMDFilters(this);
            pInfo[i].setBaseMeshTopology(bmt);
            pInfo[i].setPositionFiltering(mstateVec3d->getX());
        }
        m_pointInfo.endEdit();



        m_lineInfo.createTopologicalEngine(bmt);
#ifdef TODOTOPO
        m_lineInfo.setCreateFunction(LMDFilterLineCreationFunction);
        m_lineInfo.setCreateParameter((void *) this);
        m_lineInfo.setDestroyParameter( (void *) this );
#endif
        m_lineInfo.registerTopologicalData();

        helper::vector< LineInfo >& lInfo = *(m_lineInfo.beginEdit());
        lInfo.resize(bmt->getNbEdges());
        for (i=0; i<bmt->getNbEdges(); i++)
        {
            lInfo[i].setLMDFilters(this);
            lInfo[i].setBaseMeshTopology(bmt);
            lInfo[i].setPositionFiltering(mstateVec3d->getX());
        }
        m_lineInfo.endEdit();




        m_triangleInfo.createTopologicalEngine(bmt);
#ifdef TODOTOPO
        m_triangleInfo.setCreateFunction(LMDFilterTriangleCreationFunction);
        m_triangleInfo.setCreateParameter((void *) this);
        m_triangleInfo.setDestroyParameter( (void *) this );
#endif
        m_triangleInfo.registerTopologicalData();

        helper::vector< TriangleInfo >& tInfo = *(m_triangleInfo.beginEdit());
        tInfo.resize(bmt->getNbTriangles());
        for (i=0; i<bmt->getNbTriangles(); i++)
        {
            tInfo[i].setLMDFilters(this);
            tInfo[i].setBaseMeshTopology(bmt);
            tInfo[i].setPositionFiltering(mstateVec3d->getX());
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

    /*
        core::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();

        assert(bmt != 0);

        std::list< const core::topology::TopologyChange * >::const_iterator itBegin = bmt->beginChange();
        std::list< const core::topology::TopologyChange * >::const_iterator itEnd = bmt->endChange();

        m_pointInfo.handleTopologyEvents(itBegin, itEnd);
    	m_lineInfo.handleTopologyEvents(itBegin, itEnd);
    	m_triangleInfo.handleTopologyEvents(itBegin, itEnd);
    */
}



void TriangleLocalMinDistanceFilter::LMDFilterPointCreationFunction(unsigned int, void *param, PointInfo &pInfo, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    std::cout<<"LMDFilterPointCreationFunction is called "<<std::endl;
    const PointLocalMinDistanceFilter *pLMDFilter = static_cast< const PointLocalMinDistanceFilter * >(param);
    pInfo.setLMDFilters(pLMDFilter);
    sofa::core::topology::BaseMeshTopology * bmt = (sofa::core::topology::BaseMeshTopology *)pLMDFilter->getContext()->getTopology();
    pInfo.setBaseMeshTopology(bmt);
    /////// TODO : template de la classe
    component::container::MechanicalObject<Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<Vec3Types>*>(pLMDFilter->getContext()->getMechanicalState());
    if(pLMDFilter->isRigid())
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            pInfo.setPositionFiltering(mstateVec3d->getX0());
        }

    }
    else
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            pInfo.setPositionFiltering(mstateVec3d->getX());
        }
    }

}



void TriangleLocalMinDistanceFilter::LMDFilterLineCreationFunction(unsigned int, void *param, LineInfo &lInfo, const topology::Edge&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    std::cout<<"LMDFilterLineCreationFunction is called "<<std::endl;
    const LineLocalMinDistanceFilter *lLMDFilter = static_cast< const LineLocalMinDistanceFilter * >(param);
    lInfo.setLMDFilters(lLMDFilter);
    sofa::core::topology::BaseMeshTopology * bmt = (sofa::core::topology::BaseMeshTopology *)lLMDFilter->getContext()->getTopology();
    lInfo.setBaseMeshTopology(bmt);
    /////// TODO : template de la classe
    component::container::MechanicalObject<Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<Vec3Types>*>(lLMDFilter->getContext()->getMechanicalState());
    if(lLMDFilter->isRigid())
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            lInfo.setPositionFiltering(mstateVec3d->getX0());
        }

    }
    else
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            lInfo.setPositionFiltering(mstateVec3d->getX());
        }
    }
}



void TriangleLocalMinDistanceFilter::LMDFilterTriangleCreationFunction(unsigned int, void *param, TriangleInfo &tInfo, const topology::Triangle&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    std::cout<<"LMDFilterTriangleCreationFunction is called "<<std::endl;
    const TriangleLocalMinDistanceFilter *tLMDFilter = static_cast< const TriangleLocalMinDistanceFilter * >(param);
    tInfo.setLMDFilters(tLMDFilter);

    sofa::core::topology::BaseMeshTopology * bmt = (sofa::core::topology::BaseMeshTopology *)tLMDFilter->getContext()->getTopology();
    tInfo.setBaseMeshTopology(bmt);
    /////// TODO : template de la classe
    component::container::MechanicalObject<Vec3Types>*  mstateVec3d= dynamic_cast<component::container::MechanicalObject<Vec3Types>*>(tLMDFilter->getContext()->getMechanicalState());
    if(tLMDFilter->isRigid())
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            tInfo.setPositionFiltering(mstateVec3d->getX0());
        }

    }
    else
    {
        /////// TODO : template de la classe
        if(mstateVec3d != NULL)
        {
            tInfo.setPositionFiltering(mstateVec3d->getX());
        }
    }
}


bool TriangleLocalMinDistanceFilter::validPoint(const int pointIndex, const defaulttype::Vector3 &PQ)
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


bool TriangleLocalMinDistanceFilter::validLine(const int lineIndex, const defaulttype::Vector3 &PQ)
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


bool TriangleLocalMinDistanceFilter::validTriangle(const int triangleIndex, const defaulttype::Vector3 &PQ)
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



SOFA_DECL_CLASS(TriangleLocalMinDistanceFilter)

int TriangleLocalMinDistanceFilterClass = core::RegisterObject("This class manages Triangle collision models cones filters computations and updates.")
        .add< TriangleLocalMinDistanceFilter >()
        ;

} // namespace collision

} // namespace component

} // namespace sofa
