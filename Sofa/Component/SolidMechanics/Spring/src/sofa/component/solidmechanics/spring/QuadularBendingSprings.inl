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

#include <sofa/component/solidmechanics/spring/QuadularBendingSprings.h>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/topology/TopologyData.inl>

#include <sofa/core/topology/TopologyChange.h>

namespace sofa::component::solidmechanics::spring
{

typedef core::topology::BaseMeshTopology::Quad				Quad;
typedef core::topology::BaseMeshTopology::EdgesInQuad			EdgesInQuad;

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyEdgeCreation(Index /*edgeIndex*/, EdgeInformation &ei, const core::topology::Edge &,
        const sofa::type::vector<Index> &, const sofa::type::vector<SReal> &)
{
    unsigned int u,v;
    /// set to zero the edge stiffness matrix
    for (u=0; u<N; ++u)
    {
        for (v=0; v<N; ++v)
        {
            ei.DfDx[u][v]=0;
        }
    }

    ei.is_activated=false;
    ei.is_initialized=false;
}


template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyQuadCreation(const sofa::type::vector<Index> &quadAdded,
        const sofa::type::vector<Quad> &,
        const sofa::type::vector<sofa::type::vector<Index> > &,
        const sofa::type::vector<sofa::type::vector<SReal> > &)
{
    SReal m_ks=getKs();
    SReal m_kd=getKd();

    unsigned int u,v;

    const typename DataTypes::VecCoord& restPosition=this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    helper::WriteOnlyAccessor< Data< type::vector<EdgeInformation> > > edgeData = edgeInfo;

    for (unsigned int i=0; i<quadAdded.size(); ++i)
    {

        /// describe the jth edge index of quad no i
        EdgesInQuad te2 = this->m_topology->getEdgesInQuad(quadAdded[i]);
        /// describe the jth vertex index of quad no i
        Quad t2 = this->m_topology->getQuad(quadAdded[i]);

        for(unsigned int j=0; j<4; ++j)
        {

            EdgeInformation &ei = edgeData[te2[j]]; // ff->edgeInfo
            if(!(ei.is_initialized))
            {

                unsigned int edgeIndex = te2[j];
                ei.is_activated=true;

                /// set to zero the edge stiffness matrix
                for (u=0; u<N; ++u)
                {
                    for (v=0; v<N; ++v)
                    {
                        ei.DfDx[u][v]=0;
                    }
                }

                const auto& shell = this->m_topology->getQuadsAroundEdge(edgeIndex);
                if (shell.size()==2)
                {
                    EdgesInQuad te1;
                    Quad t1;

                    if(shell[0] == quadAdded[i])
                    {

                        te1 = this->m_topology->getEdgesInQuad(shell[1]);
                        t1 = this->m_topology->getQuad(shell[1]);

                    }
                    else   // shell[1] == quadAdded[i]
                    {

                        te1 = this->m_topology->getEdgesInQuad(shell[0]);
                        t1 = this->m_topology->getQuad(shell[0]);
                    }

                    int i1 = this->m_topology->getEdgeIndexInQuad(te1, edgeIndex); //edgeIndex //te1[j]
                    int i2 = this->m_topology->getEdgeIndexInQuad(te2, edgeIndex); // edgeIndex //te2[j]

                    ei.m1 = t1[i1]; // i1
                    ei.m2 = t2[(i2+3)%4]; // i2

                    ei.m3 = t1[(i1+3)%4]; // (i1+3)%4
                    ei.m4 = t2[i2]; // (i2+3)%4

                    //QuadularBendingSprings<DataTypes> *fftest= (QuadularBendingSprings<DataTypes> *)param;
                    ei.ks=m_ks; //(fftest->ks).getValue();
                    ei.kd=m_kd; //(fftest->kd).getValue();

                    Coord u1 = (restPosition)[ei.m1] - (restPosition)[ei.m2];
                    Real d1 = u1.norm();
                    ei.restlength1=(SReal) d1;

                    Coord u2 = (restPosition)[ei.m3] - (restPosition)[ei.m4];
                    Real d2 = u2.norm();
                    ei.restlength2=(SReal) d2;

                    ei.is_activated=true;

                }
                else
                {

                    ei.is_activated=false;

                }

                ei.is_initialized = true;
            }
        }

    }
}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyQuadDestruction(const sofa::type::vector<Index> &quadRemoved)
{
    SReal m_ks=getKs();
    SReal m_kd=getKd();

    const typename DataTypes::VecCoord& restPosition= this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    helper::WriteOnlyAccessor< Data< type::vector<EdgeInformation> > > edgeData = edgeInfo;

    for (unsigned int i=0; i<quadRemoved.size(); ++i)
    {

        /// describe the jth edge index of quad no i
        EdgesInQuad te = this->m_topology->getEdgesInQuad(quadRemoved[i]);
        /// describe the jth vertex index of quad no i
        //Quad t =  this->m_topology->getQuad(quadRemoved[i]);


        for(unsigned int j=0; j<4; ++j)
        {

            EdgeInformation &ei = edgeData[te[j]]; // ff->edgeInfo
            if(ei.is_initialized)
            {

                unsigned int edgeIndex = te[j];

                const auto& shell = this->m_topology->getQuadsAroundEdge(edgeIndex);
                if (shell.size()==3)
                {

                    EdgesInQuad te1;
                    Quad t1;
                    EdgesInQuad te2;
                    Quad t2;

                    if(shell[0] == quadRemoved[i])
                    {
                        te1 = this->m_topology->getEdgesInQuad(shell[1]);
                        t1 = this->m_topology->getQuad(shell[1]);
                        te2 = this->m_topology->getEdgesInQuad(shell[2]);
                        t2 = this->m_topology->getQuad(shell[2]);

                    }
                    else
                    {

                        if(shell[1] == quadRemoved[i])
                        {

                            te1 = this->m_topology->getEdgesInQuad(shell[2]);
                            t1 = this->m_topology->getQuad(shell[2]);
                            te2 = this->m_topology->getEdgesInQuad(shell[0]);
                            t2 = this->m_topology->getQuad(shell[0]);

                        }
                        else   // shell[2] == quadRemoved[i]
                        {

                            te1 = this->m_topology->getEdgesInQuad(shell[0]);
                            t1 = this->m_topology->getQuad(shell[0]);
                            te2 = this->m_topology->getEdgesInQuad(shell[1]);
                            t2 = this->m_topology->getQuad(shell[1]);

                        }
                    }

                    int i1 = this->m_topology->getEdgeIndexInQuad(te1, edgeIndex);
                    int i2 = this->m_topology->getEdgeIndexInQuad(te2, edgeIndex);

                    ei.m1 = t1[i1];
                    ei.m2 = t2[(i2+3)%4];

                    ei.m3 = t1[(i1+3)%4];
                    ei.m4 = t2[i2];

                    //QuadularBendingSprings<DataTypes> *fftest= (QuadularBendingSprings<DataTypes> *)param;
                    ei.ks=m_ks; //(fftest->ks).getValue();
                    ei.kd=m_kd; //(fftest->kd).getValue();

                    Coord u1 = (restPosition)[ei.m1] - (restPosition)[ei.m2];
                    Real d1 = u1.norm();
                    ei.restlength1=(SReal) d1;

                    Coord u2 = (restPosition)[ei.m3] - (restPosition)[ei.m4];
                    Real d2 = u2.norm();
                    ei.restlength2=(SReal) d2;

                    ei.is_activated=true;

                }
                else
                {

                    ei.is_activated=false;
                    ei.is_initialized = false;

                }

            }
            else
            {

                ei.is_activated=false;
                ei.is_initialized = false;

            }
        }

    }
}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyPointDestruction(const sofa::type::vector<Index> &tab)
{
    bool debug_mode = false;

    unsigned int last = this->m_topology->getNbPoints() -1;
    unsigned int i,j;

    helper::WriteOnlyAccessor< Data< type::vector<EdgeInformation> > > edgeInf = edgeInfo;

    sofa::type::vector<Index> lastIndexVec;
    for(unsigned int i_init = 0; i_init < tab.size(); ++i_init)
    {

        lastIndexVec.push_back(last - i_init);
    }

    for ( i = 0; i < tab.size(); ++i)
    {
        unsigned int i_next = i;
        bool is_reached = false;
        while( (!is_reached) && (i_next < lastIndexVec.size() - 1))
        {

            i_next += 1 ;
            is_reached = is_reached || (lastIndexVec[i_next] == tab[i]);
        }

        if(is_reached)
        {

            lastIndexVec[i_next] = lastIndexVec[i];

        }

        const auto &shell= this->m_topology->getQuadsAroundVertex(lastIndexVec[i]);
        for (j=0; j<shell.size(); ++j)
        {

            Quad tj = this->m_topology->getQuad(shell[j]);

            unsigned int vertexIndex = this->m_topology->getVertexIndexInQuad(tj, lastIndexVec[i]);

            EdgesInQuad tej = this->m_topology->getEdgesInQuad(shell[j]);

            for (unsigned int j_edge=vertexIndex; j_edge%4 !=(vertexIndex+2)%4; ++j_edge)
            {

                unsigned int ind_j = tej[j_edge%4];

                if (edgeInf[ind_j].m1 == (int) last)
                {
                    edgeInf[ind_j].m1=(int) tab[i];
                }
                else
                {
                    if (edgeInf[ind_j].m2 == (int) last)
                    {
                        edgeInf[ind_j].m2=(int) tab[i];
                    }
                }

                if (edgeInf[ind_j].m3 == (int) last)
                {
                    edgeInf[ind_j].m3=(int) tab[i];
                }
                else
                {
                    if (edgeInf[ind_j].m4 == (int) last)
                    {
                        edgeInf[ind_j].m4=(int) tab[i];
                    }
                }

            }
        }

        if(debug_mode)
        {
            for (unsigned int j_loc=0; j_loc<edgeInf.size(); ++j_loc)
            {

                //bool is_forgotten = false;
                if (edgeInf[j_loc].m1 == (int) last)
                {
                    edgeInf[j_loc].m1 =(int) tab[i];
                    //is_forgotten=true;
                }
                else
                {
                    if (edgeInf[j_loc].m2 ==(int) last)
                    {
                        edgeInf[j_loc].m2 =(int) tab[i];
                        //is_forgotten=true;
                    }

                }

                if (edgeInf[j_loc].m3 == (int) last)
                {
                    edgeInf[j_loc].m3 =(int) tab[i];
                    //is_forgotten=true;
                }
                else
                {
                    if (edgeInf[j_loc].m4 ==(int) last)
                    {
                        edgeInf[j_loc].m4 =(int) tab[i];
                        //is_forgotten=true;
                    }

                }
            }
        }

        --last;
    }
}



template< class DataTypes>
void QuadularBendingSprings<DataTypes>::applyPointRenumbering(const sofa::type::vector<Index> &tab)
{
    helper::WriteOnlyAccessor< Data< type::vector<EdgeInformation> > > edgeInf = edgeInfo;
    for (unsigned  int i = 0; i < this->m_topology->getNbEdges(); ++i)
    {
        if(edgeInf[i].is_activated)
        {
            edgeInf[i].m1  = tab[edgeInf[i].m1];
            edgeInf[i].m2  = tab[edgeInf[i].m2];
            edgeInf[i].m3  = tab[edgeInf[i].m3];
            edgeInf[i].m4  = tab[edgeInf[i].m4];
        }
    }
}


template<class DataTypes>
QuadularBendingSprings<DataTypes>::QuadularBendingSprings()
    : f_ks ( initData(&f_ks,(SReal) 100000.0,"stiffness","uniform stiffness for the all springs"))
    , f_kd ( initData(&f_kd,(SReal) 1.0,"damping","uniform damping for the all springs"))
    , l_topology(initLink("topology", "link to the topology container"))
    , edgeInfo ( initData(&edgeInfo, "edgeInfo","Internal edge data"))
    , m_topology(nullptr)
    , updateMatrix(true)
{

}


template<class DataTypes>
QuadularBendingSprings<DataTypes>::~QuadularBendingSprings()
{

}


template<class DataTypes>
void QuadularBendingSprings<DataTypes>::init()
{
    this->Inherited::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (m_topology->getNbQuads()==0)
    {
        msg_warning() << "No Quads found in linked Topology.";
    }

    edgeInfo.createTopologyHandler(m_topology);
    edgeInfo.linkToPointDataArray();
    edgeInfo.linkToQuadDataArray();

    /// prepare to store info in the edge array
    helper::WriteOnlyAccessor< Data< type::vector<EdgeInformation> > > edgeInf = edgeInfo;
    edgeInf.resize(m_topology->getNbEdges());

    // set edge tensor to 0
    for (Index i=0; i<m_topology->getNbEdges(); ++i)
    {
        applyEdgeCreation(i, edgeInf[i],
            m_topology->getEdge(i),  (const sofa::type::vector< Index > )0,
            (const sofa::type::vector< SReal >)0);
    }

    // create edge tensor by calling the quad creation function
    sofa::type::vector<Index> quadAdded;
    for (unsigned int i=0; i<m_topology->getNbQuads(); ++i)
        quadAdded.push_back(i);

    applyQuadCreation(quadAdded,
        (const sofa::type::vector<Quad>)0,
        (const sofa::type::vector<sofa::type::vector<Index> >)0,
        (const sofa::type::vector<sofa::type::vector<SReal> >)0);

    edgeInfo.setCreationCallback([this](Index edgeIndex, EdgeInformation& ei,
        const core::topology::BaseMeshTopology::Edge& edge,
        const sofa::type::vector< Index >& ancestors,
        const sofa::type::vector< SReal >& coefs)
    {
        applyEdgeCreation(edgeIndex, ei, edge, ancestors, coefs);
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSADDED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::QuadsAdded* quadAdd = static_cast<const core::topology::QuadsAdded*>(eventTopo);
        applyQuadCreation(quadAdd->getIndexArray(), quadAdd->getElementArray(), quadAdd->ancestorsList, quadAdd->coefs);
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::QUADSREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::QuadsRemoved* quadRemove = static_cast<const core::topology::QuadsRemoved*>(eventTopo);
        applyQuadDestruction(quadRemove->getArray());
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::POINTSREMOVED, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::PointsRemoved* pRemove = static_cast<const core::topology::PointsRemoved*>(eventTopo);
        applyPointDestruction(pRemove->getArray());
    });

    edgeInfo.addTopologyEventCallBack(sofa::core::topology::TopologyChangeType::POINTSRENUMBERING, [this](const core::topology::TopologyChange* eventTopo) {
        const core::topology::PointsRenumbering* pRenum = static_cast<const core::topology::PointsRenumbering*>(eventTopo);
        applyPointRenumbering(pRenum->getIndexArray());
    });

}

template <class DataTypes>
SReal QuadularBendingSprings<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* d_x */) const
{
    msg_error() << "getPotentialEnergy-not-implemented !!!";
    return 0;
}

template<class DataTypes>
void QuadularBendingSprings<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    size_t nbEdges=m_topology->getNbEdges();

    EdgeInformation *einfo;

    type::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());

    //const type::vector<Spring>& m_springs= this->springs.getValue();
    //this->dfdx.resize(nbEdges); //m_springs.size()
    f.resize(x.size());
    m_potentialEnergy = 0;

    for(unsigned int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];

        if(einfo->is_activated)
        {
            //this->addSpringForce(m_potentialEnergy,f,x,v, i, einfo->spring);

            int a1 = einfo->m1;
            int b1 = einfo->m2;
            int a2 = einfo->m3;
            int b2 = einfo->m4;
            Coord u1 = x[b1]-x[a1];
            Real d1 = u1.norm();
            Coord u2 = x[b2]-x[a2];
            Real d2 = u2.norm();
            if( d1>1.0e-4 )
            {
                Real inverseLength = 1.0f/d1;
                u1 *= inverseLength;
                Real elongation = (Real)(d1 - einfo->restlength1);
                m_potentialEnergy += elongation * elongation * einfo->ks / 2;

                Deriv relativeVelocity = v[b1]-v[a1];
                Real elongationVelocity = dot(u1,relativeVelocity);
                Real forceIntensity = (Real)(einfo->ks*elongation+einfo->kd*elongationVelocity);
                Deriv force = u1*forceIntensity;
                f[a1]+=force;
                f[b1]-=force;

                updateMatrix=true;

                Mat& m = einfo->DfDx; //Mat& m = this->dfdx[i];
                Real tgt = forceIntensity * inverseLength;
                for( int j=0; j<N; ++j )
                {
                    for( int k=0; k<N; ++k )
                    {
                        m[j][k] = ((Real)einfo->ks-tgt) * u1[j] * u1[k];
                    }
                    m[j][j] += tgt;
                }
            }
            else // null length, no force and no stiffness
            {
                Mat& m = einfo->DfDx; //Mat& m = this->dfdx[i];
                for( int j=0; j<N; ++j )
                {
                    for( int k=0; k<N; ++k )
                    {
                        m[j][k] = 0;
                    }
                }
            }

            if( d2>1.0e-4 )
            {
                Real inverseLength = 1.0f/d2;
                u2 *= inverseLength;
                Real elongation = (Real)(d2 - einfo->restlength2);
                m_potentialEnergy += elongation * elongation * einfo->ks / 2;

                Deriv relativeVelocity = v[b2]-v[a2];
                Real elongationVelocity = dot(u2,relativeVelocity);
                Real forceIntensity = (Real)(einfo->ks*elongation+einfo->kd*elongationVelocity);
                Deriv force = u2*forceIntensity;
                f[a2]+=force;
                f[b2]-=force;

                updateMatrix=true;

                Mat& m = einfo->DfDx; //Mat& m = this->dfdx[i];
                Real tgt = forceIntensity * inverseLength;
                for( int j=0; j<N; ++j )
                {
                    for( int k=0; k<N; ++k )
                    {
                        m[j][k] = ((Real)einfo->ks-tgt) * u2[j] * u2[k];
                    }
                    m[j][j] += tgt;
                }
            }
            else // null length, no force and no stiffness
            {
                Mat& m = einfo->DfDx; //Mat& m = this->dfdx[i];
                for( int j=0; j<N; ++j )
                {
                    for( int k=0; k<N; ++k )
                    {
                        m[j][k] = 0;
                    }
                }
            }

        }
    }

    edgeInfo.endEdit();
    d_f.endEdit();

    //for (unsigned int i=0; i<springs.size(); i++)
    //{
    //    this->addSpringForce(m_potentialEnergy,f,x,v, i, springs[i]);
    //}
}

template<class DataTypes>
void QuadularBendingSprings<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    size_t nbEdges=m_topology->getNbEdges();

    const EdgeInformation *einfo;

    const type::vector<EdgeInformation>& edgeInf = edgeInfo.getValue();

    df.resize(dx.size());
    //const type::vector<Spring>& springs = this->springs.getValue();

    for(unsigned int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];

        if(einfo->is_activated)
        {
            //this->addSpringDForce(df,dx, i, einfo->spring);

            const int a1 = einfo->m1;
            const int b1 = einfo->m2;
            const Coord d1 = dx[b1]-dx[a1];
            const int a2 = einfo->m3;
            const int b2 = einfo->m4;
            const Coord d2 = dx[b2]-dx[a2];
            const Deriv dforce1 = (einfo->DfDx*d1) * kFactor;
            const Deriv dforce2 = (einfo->DfDx*d2) * kFactor;
            df[a1]+=dforce1;
            df[b1]-=dforce1;
            df[a2]+=dforce2;
            df[b2]-=dforce2;

            updateMatrix=false;
        }
    }
    d_df.endEdit();

}

template<class DataTypes>
void QuadularBendingSprings<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    vparams->drawTool()->disableLighting();

    const type::vector<EdgeInformation>& edgeInf = edgeInfo.getValue();
    std::vector<sofa::type::Vec3> vertices;
    std::vector<sofa::type::RGBAColor> colors;
    constexpr sofa::type::RGBAColor green_color = sofa::type::RGBAColor::green();
    constexpr sofa::type::RGBAColor red_color   = sofa::type::RGBAColor::red();
    constexpr sofa::type::RGBAColor color1 = sofa::type::RGBAColor(1,0.5, 0,1);
    constexpr sofa::type::RGBAColor color2 = sofa::type::RGBAColor(0,1,0.5,1);

    for(unsigned int i=0; i<edgeInf.size(); ++i)
    {
        if(edgeInf[i].is_activated)
        {
            bool external=true;
            Real d1 = (x[edgeInf[i].m2]-x[edgeInf[i].m1]).norm();
            if (external)
            {
                if (d1<edgeInf[i].restlength2*0.9999)
                    colors.push_back(red_color);
                else
                    colors.push_back(green_color);
            }
            else
            {
                if (d1<edgeInf[i].restlength1*0.9999)
                    colors.push_back(color1);
                else
                    colors.push_back(color2);
            }

            vertices.push_back( x[edgeInf[i].m1] );
            vertices.push_back( x[edgeInf[i].m2] );

            Real d2 = (x[edgeInf[i].m4]-x[edgeInf[i].m3]).norm();
            if (external)
            {
                if (d2<edgeInf[i].restlength2*0.9999)
                    colors.push_back(red_color);
                else
                    colors.push_back(green_color);
            }
            else
            {
                if (d2<edgeInf[i].restlength2*0.9999)
                    colors.push_back(color1);
                else
                    colors.push_back(color2);
            }

            vertices.push_back( x[edgeInf[i].m3] );
            vertices.push_back( x[edgeInf[i].m4] );
        }
    }
    vparams->drawTool()->drawLines(vertices, 1, colors);

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, false);


    vertices.clear();
    for(unsigned int i=0; i<m_topology->getNbQuads(); ++i)
    {
        for(unsigned int j = 0 ; j<4 ; j++)
            vertices.push_back(x[m_topology->getQuad(i)[j]]);
    }
    vparams->drawTool()->drawQuads(vertices, sofa::type::RGBAColor::red());


}


} // namespace sofa::component::solidmechanics::spring
