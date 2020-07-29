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
#ifndef SOFA_COMPONENT_FORCEFIELD_QUADULARBENDINGSPRINGS_INL
#define SOFA_COMPONENT_FORCEFIELD_QUADULARBENDINGSPRINGS_INL

#include <SofaGeneralDeformable/QuadularBendingSprings.h>
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <sofa/defaulttype/RGBAColor.h>
#include <SofaBaseTopology/TopologyData.inl>

#include <sofa/core/topology/TopologyChange.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

typedef core::topology::BaseMeshTopology::Quad				Quad;
typedef core::topology::BaseMeshTopology::EdgesInQuad			EdgesInQuad;

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::EdgeBSHandler::applyCreateFunction(unsigned int /*edgeIndex*/, EdgeInformation &ei, const core::topology::Edge &,
        const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (ff)
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
}


template< class DataTypes>
void QuadularBendingSprings<DataTypes>::EdgeBSHandler::applyQuadCreation(const sofa::helper::vector<unsigned int> &quadAdded,
        const sofa::helper::vector<Quad> &,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &,
        const sofa::helper::vector<sofa::helper::vector<double> > &)
{

    if (ff)
    {

        double m_ks=ff->getKs();
        double m_kd=ff->getKd();

        unsigned int u,v;

        unsigned int nb_activated = 0;

        const typename DataTypes::VecCoord& restPosition=ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

        helper::vector<EdgeInformation>& edgeData = *(ff->edgeInfo.beginEdit());

        for (unsigned int i=0; i<quadAdded.size(); ++i)
        {

            /// describe the jth edge index of quad no i
            EdgesInQuad te2 = ff->m_topology->getEdgesInQuad(quadAdded[i]);
            /// describe the jth vertex index of quad no i
            Quad t2 = ff->m_topology->getQuad(quadAdded[i]);

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

                    const sofa::helper::vector< unsigned int > shell = ff->m_topology->getQuadsAroundEdge(edgeIndex);
                    if (shell.size()==2)
                    {

                        nb_activated+=1;

                        EdgesInQuad te1;
                        Quad t1;

                        if(shell[0] == quadAdded[i])
                        {

                            te1 = ff->m_topology->getEdgesInQuad(shell[1]);
                            t1 =  ff->m_topology->getQuad(shell[1]);

                        }
                        else   // shell[1] == quadAdded[i]
                        {

                            te1 = ff->m_topology->getEdgesInQuad(shell[0]);
                            t1 =  ff->m_topology->getQuad(shell[0]);
                        }

                        int i1 = ff->m_topology->getEdgeIndexInQuad(te1, edgeIndex); //edgeIndex //te1[j]
                        int i2 = ff->m_topology->getEdgeIndexInQuad(te2, edgeIndex); // edgeIndex //te2[j]

                        ei.m1 = t1[i1]; // i1
                        ei.m2 = t2[(i2+3)%4]; // i2

                        ei.m3 = t1[(i1+3)%4]; // (i1+3)%4
                        ei.m4 = t2[i2]; // (i2+3)%4

                        //QuadularBendingSprings<DataTypes> *fftest= (QuadularBendingSprings<DataTypes> *)param;
                        ei.ks=m_ks; //(fftest->ks).getValue();
                        ei.kd=m_kd; //(fftest->kd).getValue();

                        Coord u1 = (restPosition)[ei.m1] - (restPosition)[ei.m2];
                        Real d1 = u1.norm();
                        ei.restlength1=(double) d1;

                        Coord u2 = (restPosition)[ei.m3] - (restPosition)[ei.m4];
                        Real d2 = u2.norm();
                        ei.restlength2=(double) d2;

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
        ff->edgeInfo.endEdit();
    }

}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::EdgeBSHandler::applyQuadDestruction(const sofa::helper::vector<unsigned int> &quadRemoved)
{
    if (ff)
    {

        double m_ks=ff->getKs(); // typename DataTypes::
        double m_kd=ff->getKd(); // typename DataTypes::

        //unsigned int u,v;

        const typename DataTypes::VecCoord& restPosition=ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        helper::vector<EdgeInformation>& edgeData = *(ff->edgeInfo.beginEdit());

        for (unsigned int i=0; i<quadRemoved.size(); ++i)
        {

            /// describe the jth edge index of quad no i
            EdgesInQuad te = ff->m_topology->getEdgesInQuad(quadRemoved[i]);
            /// describe the jth vertex index of quad no i
            //Quad t =  ff->m_topology->getQuad(quadRemoved[i]);


            for(unsigned int j=0; j<4; ++j)
            {

                EdgeInformation &ei = edgeData[te[j]]; // ff->edgeInfo
                if(ei.is_initialized)
                {

                    unsigned int edgeIndex = te[j];

                    const sofa::helper::vector< unsigned int > shell = ff->m_topology->getQuadsAroundEdge(edgeIndex);
                    if (shell.size()==3)
                    {

                        EdgesInQuad te1;
                        Quad t1;
                        EdgesInQuad te2;
                        Quad t2;

                        if(shell[0] == quadRemoved[i])
                        {
                            te1 = ff->m_topology->getEdgesInQuad(shell[1]);
                            t1 =  ff->m_topology->getQuad(shell[1]);
                            te2 = ff->m_topology->getEdgesInQuad(shell[2]);
                            t2 =  ff->m_topology->getQuad(shell[2]);

                        }
                        else
                        {

                            if(shell[1] == quadRemoved[i])
                            {

                                te1 = ff->m_topology->getEdgesInQuad(shell[2]);
                                t1 =  ff->m_topology->getQuad(shell[2]);
                                te2 = ff->m_topology->getEdgesInQuad(shell[0]);
                                t2 =  ff->m_topology->getQuad(shell[0]);

                            }
                            else   // shell[2] == quadRemoved[i]
                            {

                                te1 = ff->m_topology->getEdgesInQuad(shell[0]);
                                t1 =  ff->m_topology->getQuad(shell[0]);
                                te2 = ff->m_topology->getEdgesInQuad(shell[1]);
                                t2 =  ff->m_topology->getQuad(shell[1]);

                            }
                        }

                        int i1 = ff->m_topology->getEdgeIndexInQuad(te1, edgeIndex);
                        int i2 = ff->m_topology->getEdgeIndexInQuad(te2, edgeIndex);

                        ei.m1 = t1[i1];
                        ei.m2 = t2[(i2+3)%4];

                        ei.m3 = t1[(i1+3)%4];
                        ei.m4 = t2[i2];

                        //QuadularBendingSprings<DataTypes> *fftest= (QuadularBendingSprings<DataTypes> *)param;
                        ei.ks=m_ks; //(fftest->ks).getValue();
                        ei.kd=m_kd; //(fftest->kd).getValue();

                        Coord u1 = (restPosition)[ei.m1] - (restPosition)[ei.m2];
                        Real d1 = u1.norm();
                        ei.restlength1=(double) d1;

                        Coord u2 = (restPosition)[ei.m3] - (restPosition)[ei.m4];
                        Real d2 = u2.norm();
                        ei.restlength2=(double) d2;

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
        ff->edgeInfo.endEdit();
    }
}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::EdgeBSHandler::ApplyTopologyChange(const core::topology::QuadsAdded* e)
{
    const sofa::helper::vector<unsigned int> &quadAdded = e->getIndexArray();
    const sofa::helper::vector<Quad> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyQuadCreation(quadAdded, elems, ancestors, coefs);
}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::EdgeBSHandler::ApplyTopologyChange(const core::topology::QuadsRemoved* e)
{
    const sofa::helper::vector<unsigned int> &quadRemoved = e->getArray();

    applyQuadDestruction(quadRemoved);
}


template< class DataTypes>
void QuadularBendingSprings<DataTypes>::EdgeBSHandler::applyPointDestruction(const sofa::helper::vector<unsigned int> &tab)
{
    if(ff)
    {
        bool debug_mode = false;

        unsigned int last = ff->m_topology->getNbPoints() -1;
        unsigned int i,j;

        helper::vector<EdgeInformation>& edgeInf = *(ff->edgeInfo.beginEdit());

        sofa::helper::vector<unsigned int> lastIndexVec;
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

            const sofa::helper::vector<unsigned int> &shell= ff->m_topology->getQuadsAroundVertex(lastIndexVec[i]);
            for (j=0; j<shell.size(); ++j)
            {

                Quad tj = ff->m_topology->getQuad(shell[j]);

                unsigned int vertexIndex = ff->m_topology->getVertexIndexInQuad(tj, lastIndexVec[i]);

                EdgesInQuad tej = ff->m_topology->getEdgesInQuad(shell[j]);

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
        ff->edgeInfo.endEdit();
    }
}



template< class DataTypes>
void QuadularBendingSprings<DataTypes>::EdgeBSHandler::applyPointRenumbering(const sofa::helper::vector<unsigned int> &tab)
{
    if(ff)
    {
        helper::vector<EdgeInformation>& edgeInf = *(ff->edgeInfo.beginEdit());
        for (unsigned  int i = 0; i < ff->m_topology->getNbEdges(); ++i)
        {
            if(edgeInf[i].is_activated)
            {
                edgeInf[i].m1  = tab[edgeInf[i].m1];
                edgeInf[i].m2  = tab[edgeInf[i].m2];
                edgeInf[i].m3  = tab[edgeInf[i].m3];
                edgeInf[i].m4  = tab[edgeInf[i].m4];
            }
        }
        ff->edgeInfo.endEdit();
    }
}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::EdgeBSHandler::ApplyTopologyChange(const core::topology::PointsRemoved* e)
{
    const sofa::helper::vector<unsigned int> & tab = e->getArray();
    applyPointDestruction(tab);
}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::EdgeBSHandler::ApplyTopologyChange(const core::topology::PointsRenumbering* e)
{
    const sofa::helper::vector<unsigned int> &newIndices = e->getIndexArray();
    applyPointRenumbering(newIndices);
}



template<class DataTypes>
QuadularBendingSprings<DataTypes>::QuadularBendingSprings()
    : f_ks ( initData(&f_ks,(double) 100000.0,"stiffness","uniform stiffness for the all springs"))
    , f_kd ( initData(&f_kd,(double) 1.0,"damping","uniform damping for the all springs"))
    , l_topology(initLink("topology", "link to the topology container"))
    , edgeInfo ( initData(&edgeInfo, "edgeInfo","Internal edge data"))
    , m_topology(nullptr)
    , updateMatrix(true)
{
    edgeHandler = new EdgeBSHandler(this, &edgeInfo);
}


template<class DataTypes>
QuadularBendingSprings<DataTypes>::~QuadularBendingSprings()
{
    if(edgeHandler)
        delete edgeHandler;
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
        sofa::core::objectmodel::BaseObject::d_componentstate.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (m_topology->getNbQuads()==0)
    {
        msg_warning() << "No Quads found in linked Topology.";
    }

    edgeInfo.createTopologicalEngine(m_topology,edgeHandler);
    edgeInfo.linkToPointDataArray();
    edgeInfo.linkToQuadDataArray();
    edgeInfo.registerTopologicalData();

    /// prepare to store info in the edge array
    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());
    edgeInf.resize(m_topology->getNbEdges());

    // set edge tensor to 0
    for (unsigned int i=0; i<m_topology->getNbEdges(); ++i)
    {
        edgeHandler->applyCreateFunction(i, edgeInf[i],
                m_topology->getEdge(i),  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }

    // create edge tensor by calling the quad creation function
    sofa::helper::vector<unsigned int> quadAdded;
    for (unsigned int i=0; i<m_topology->getNbQuads(); ++i)
        quadAdded.push_back(i);

    edgeHandler->applyQuadCreation(quadAdded,
            (const sofa::helper::vector<Quad>)0,
            (const sofa::helper::vector<sofa::helper::vector<unsigned int> >)0,
            (const sofa::helper::vector<sofa::helper::vector<double> >)0);


    edgeInfo.endEdit();
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

    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());

    //const helper::vector<Spring>& m_springs= this->springs.getValue();
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
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    size_t nbEdges=m_topology->getNbEdges();

    const EdgeInformation *einfo;

    const helper::vector<EdgeInformation>& edgeInf = edgeInfo.getValue();

    df.resize(dx.size());
    //const helper::vector<Spring>& springs = this->springs.getValue();

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

    vparams->drawTool()->saveLastState();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    vparams->drawTool()->disableLighting();

    const helper::vector<EdgeInformation>& edgeInf = edgeInfo.getValue();
    std::vector<sofa::defaulttype::Vector3> vertices;
    std::vector<sofa::defaulttype::Vec4f> colors;
    sofa::defaulttype::RGBAColor green_color = sofa::defaulttype::RGBAColor::green();
    sofa::defaulttype::RGBAColor red_color   = sofa::defaulttype::RGBAColor::red();
    sofa::defaulttype::RGBAColor color1 = sofa::defaulttype::RGBAColor(1,0.5, 0,1);
    sofa::defaulttype::RGBAColor color2 = sofa::defaulttype::RGBAColor(0,1,0.5,1);

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
    vparams->drawTool()->drawQuads(vertices, sofa::defaulttype::RGBAColor::red());

    vparams->drawTool()->restoreLastState();
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_QUADULARBENDINGSPRINGS_INL
