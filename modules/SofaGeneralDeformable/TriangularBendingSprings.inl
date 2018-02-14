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
//
// C++ Implementation: TriangularBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_INL
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_INL

#include <SofaGeneralDeformable/TriangularBendingSprings.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging

#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/TopologyData.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{

typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;

template< class DataTypes>
void TriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyCreateFunction(unsigned int , EdgeInformation &ei, const core::topology::Edge &, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
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
void TriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyTriangleCreation(const sofa::helper::vector<unsigned int> &triangleAdded, const sofa::helper::vector<core::topology::Triangle> &, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &, const sofa::helper::vector<sofa::helper::vector<double> > &)
{
    using namespace core::topology;
    if (ff)
    {

        double m_ks=ff->getKs();
        double m_kd=ff->getKd();

        unsigned int u,v;

        unsigned int nb_activated = 0;

        const typename DataTypes::VecCoord& restPosition=ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

        helper::vector<EdgeInformation>& edgeData = *(ff->edgeInfo.beginEdit());

        for (unsigned int i=0; i<triangleAdded.size(); ++i)
        {

            /// describe the jth edge index of triangle no i
            EdgesInTriangle te2 = ff->_topology->getEdgesInTriangle(triangleAdded[i]);
            /// describe the jth vertex index of triangle no i
            Triangle t2 = ff->_topology->getTriangle(triangleAdded[i]);

            for(unsigned int j=0; j<3; ++j)
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

                    const sofa::helper::vector< unsigned int > shell = ff->_topology->getTrianglesAroundEdge(edgeIndex);
                    if (shell.size()==2)
                    {

                        nb_activated+=1;

                        EdgesInTriangle te1;
                        Triangle t1;

                        if(shell[0] == triangleAdded[i])
                        {

                            te1 = ff->_topology->getEdgesInTriangle(shell[1]);
                            t1 = ff->_topology->getTriangle(shell[1]);

                        }
                        else   // shell[1] == triangleAdded[i]
                        {

                            te1 = ff->_topology->getEdgesInTriangle(shell[0]);
                            t1 = ff->_topology->getTriangle(shell[0]);
                        }

                        int i1 = ff->_topology->getEdgeIndexInTriangle(te1, edgeIndex); //edgeIndex //te1[j]
                        int i2 = ff->_topology->getEdgeIndexInTriangle(te2, edgeIndex); // edgeIndex //te2[j]

                        ei.m1 = t1[i1];
                        ei.m2 = t2[i2];

                        //TriangularBendingSprings<DataTypes> *fftest= (TriangularBendingSprings<DataTypes> *)param;
                        ei.ks=m_ks; //(fftest->ks).getValue();
                        ei.kd=m_kd; //(fftest->kd).getValue();

                        Coord u = (restPosition)[ei.m1] - (restPosition)[ei.m2];

                        Real d = u.norm();

                        ei.restlength=(double) d;

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
void TriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyTriangleDestruction(const sofa::helper::vector<unsigned int> &triangleRemoved)
{
    using namespace core::topology;
    if (ff)
    {

        double m_ks=ff->getKs(); // typename DataTypes::
        double m_kd=ff->getKd(); // typename DataTypes::

        //unsigned int u,v;

        const typename DataTypes::VecCoord& restPosition=ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        helper::vector<EdgeInformation>& edgeData = *(ff->edgeInfo.beginEdit());

        for (unsigned int i=0; i<triangleRemoved.size(); ++i)
        {
            /// describe the jth edge index of triangle no i
            EdgesInTriangle te = ff->_topology->getEdgesInTriangle(triangleRemoved[i]);
            /// describe the jth vertex index of triangle no i
            //Triangle t = ff->_topology->getTriangle(triangleRemoved[i]);


            for(unsigned int j=0; j<3; ++j)
            {

                EdgeInformation &ei = edgeData[te[j]]; // ff->edgeInfo
                if(ei.is_initialized)
                {

                    unsigned int edgeIndex = te[j];

                    const sofa::helper::vector< unsigned int > shell = ff->_topology->getTrianglesAroundEdge(edgeIndex);
                    if (shell.size()==3)
                    {

                        EdgesInTriangle te1;
                        Triangle t1;
                        EdgesInTriangle te2;
                        Triangle t2;

                        if(shell[0] == triangleRemoved[i])
                        {
                            te1 = ff->_topology->getEdgesInTriangle(shell[1]);
                            t1 = ff->_topology->getTriangle(shell[1]);
                            te2 = ff->_topology->getEdgesInTriangle(shell[2]);
                            t2 = ff->_topology->getTriangle(shell[2]);

                        }
                        else
                        {

                            if(shell[1] == triangleRemoved[i])
                            {

                                te1 = ff->_topology->getEdgesInTriangle(shell[2]);
                                t1 = ff->_topology->getTriangle(shell[2]);
                                te2 = ff->_topology->getEdgesInTriangle(shell[0]);
                                t2 = ff->_topology->getTriangle(shell[0]);

                            }
                            else   // shell[2] == triangleRemoved[i]
                            {

                                te1 = ff->_topology->getEdgesInTriangle(shell[0]);
                                t1 = ff->_topology->getTriangle(shell[0]);
                                te2 = ff->_topology->getEdgesInTriangle(shell[1]);
                                t2 = ff->_topology->getTriangle(shell[1]);

                            }
                        }

                        int i1 = ff->_topology->getEdgeIndexInTriangle(te1, edgeIndex);
                        int i2 = ff->_topology->getEdgeIndexInTriangle(te2, edgeIndex);

                        ei.m1 = t1[i1];
                        ei.m2 = t2[i2];

                        //TriangularBendingSprings<DataTypes> *fftest= (TriangularBendingSprings<DataTypes> *)param;
                        ei.ks=m_ks; //(fftest->ks).getValue();
                        ei.kd=m_kd; //(fftest->kd).getValue();

                        Coord u = (restPosition)[ei.m1] - (restPosition)[ei.m2];
                        Real d = u.norm();

                        ei.restlength=(double) d;

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

template<class DataTypes>
void TriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::ApplyTopologyChange(const core::topology::TrianglesAdded* e)
{
    using namespace core::topology;
    const sofa::helper::vector<unsigned int> &triangleAdded = e->getIndexArray();
    const sofa::helper::vector<Triangle> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTriangleCreation(triangleAdded, elems, ancestors, coefs);
}

template<class DataTypes>
void TriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::ApplyTopologyChange(const core::topology::TrianglesRemoved* e)
{
    const sofa::helper::vector<unsigned int> &triangleRemoved = e->getArray();

    applyTriangleDestruction(triangleRemoved);
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyPointDestruction(const sofa::helper::vector<unsigned int> &tab)
{
    using namespace core::topology;
    if(ff)
    {
        bool debug_mode = false;

        unsigned int last = ff->_topology->getNbPoints() -1;
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

            const sofa::helper::vector<unsigned int> &shell= ff->_topology->getTrianglesAroundVertex(lastIndexVec[i]);
            for (j=0; j<shell.size(); ++j)
            {

                Triangle tj = ff->_topology->getTriangle(shell[j]);

                int vertexIndex = ff->_topology->getVertexIndexInTriangle(tj, lastIndexVec[i]);

                EdgesInTriangle tej = ff->_topology->getEdgesInTriangle(shell[j]);

                unsigned int ind_j = tej[vertexIndex];

                if (edgeInf[ind_j].m1 == (int) last)
                {
                    edgeInf[ind_j].m1=(int) tab[i];
                    //sout << "INFO_print : OK m1 for ind_j =" << ind_j << sendl;
                }
                else
                {
                    if (edgeInf[ind_j].m2 == (int) last)
                    {
                        edgeInf[ind_j].m2=(int) tab[i];
                        //sout << "INFO_print : OK m2 for ind_j =" << ind_j << sendl;
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
                        //sout << "INFO_print : TriangularBendingSprings - MISS m1 for j_loc =" << j_loc << sendl;

                    }
                    else
                    {
                        if (edgeInf[j_loc].m2 ==(int) last)
                        {
                            edgeInf[j_loc].m2 =(int) tab[i];
                            //is_forgotten=true;
                            //sout << "INFO_print : TriangularBendingSprings - MISS m2 for j_loc =" << j_loc << sendl;

                        }

                    }

                }
            }

            --last;
        }

        ff->edgeInfo.endEdit();
    }
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::applyPointRenumbering(const sofa::helper::vector<unsigned int> &tab)
{
    if(ff)
    {
        helper::vector<EdgeInformation>& edgeInf = *(ff->edgeInfo.beginEdit());
        for (int i = 0; i < ff->_topology->getNbEdges(); ++i)
        {
            if(edgeInf[i].is_activated)
            {
                edgeInf[i].m1  = tab[edgeInf[i].m1];
                edgeInf[i].m2  = tab[edgeInf[i].m2];
            }
        }
        ff->edgeInfo.endEdit();
    }
}

template<class DataTypes>
void TriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::ApplyTopologyChange(const core::topology::PointsRemoved* e)
{
    const sofa::helper::vector<unsigned int> & tab = e->getArray();
    applyPointDestruction(tab);
}

template<class DataTypes>
void TriangularBendingSprings<DataTypes>::TriangularBSEdgeHandler::ApplyTopologyChange(const core::topology::PointsRenumbering* e)
{
    const sofa::helper::vector<unsigned int> &newIndices = e->getIndexArray();
    applyPointRenumbering(newIndices);
}


template<class DataTypes>
TriangularBendingSprings<DataTypes>::TriangularBendingSprings(/*double _ks, double _kd*/)
    : edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
    , updateMatrix(true)
    , f_ks(initData(&f_ks,(double) 100000.0,"stiffness","uniform stiffness for the all springs")) //(Real)0.3 ??
    , f_kd(initData(&f_kd,(double) 1.0,"damping","uniform damping for the all springs")) // (Real)1000. ??
    , edgeHandler(NULL)
{
    // Create specific handler for EdgeData
    edgeHandler = new TriangularBSEdgeHandler(this, &edgeInfo);
    //serr<<"TriangularBendingSprings<DataTypes>::TriangularBendingSprings"<<sendl;
}

template<class DataTypes>
TriangularBendingSprings<DataTypes>::~TriangularBendingSprings()
{
    if(edgeHandler) delete edgeHandler;
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::init()
{
    //serr << "initializing TriangularBendingSprings" << sendl;
    this->Inherited::init();

    _topology = this->getContext()->getMeshTopology();

    if (_topology->getNbTriangles()==0)
    {
        serr << "ERROR(TriangularBendingSprings): object must have a Triangular Set Topology."<<sendl;
        return;
    }
    edgeInfo.createTopologicalEngine(_topology,edgeHandler);
    edgeInfo.linkToPointDataArray();
    edgeInfo.linkToTriangleDataArray();
    edgeInfo.registerTopologicalData();

    this->reinit();
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::reinit()
{
    using namespace core::topology;
    /// prepare to store info in the edge array
    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());
    edgeInf.resize(_topology->getNbEdges());
    int i;
    // set edge tensor to 0
    for (i=0; i<_topology->getNbEdges(); ++i)
    {

        edgeHandler->applyCreateFunction(i, edgeInf[i],
                _topology->getEdge(i),  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }

    // create edge tensor by calling the triangle creation function
    sofa::helper::vector<unsigned int> triangleAdded;
    for (i=0; i<_topology->getNbTriangles(); ++i)
        triangleAdded.push_back(i);

    edgeHandler->applyTriangleCreation(triangleAdded,
            (const sofa::helper::vector<Triangle>)0,
            (const sofa::helper::vector<sofa::helper::vector<unsigned int> >)0,
            (const sofa::helper::vector<sofa::helper::vector<double> >)0);

    edgeInfo.endEdit();
}

template <class DataTypes>
SReal TriangularBendingSprings<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* d_x */) const
{
    serr<<"TriangularBendingSprings::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    int nbEdges=_topology->getNbEdges();
    EdgeInformation *einfo;
    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());

    //const helper::vector<Spring>& m_springs= this->springs.getValue();
    //this->dfdx.resize(nbEdges); //m_springs.size()
    f.resize(x.size());
    m_potentialEnergy = 0;
    /*        serr<<"TriangularBendingSprings<DataTypes>::addForce()"<<sendl;*/

#if 0
    const VecCoord& x_rest = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
#endif

    for(int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];

        // safety check
#if 0
        {
            EdgeInformation e2;
            const sofa::helper::vector< unsigned int > shell = _topology->getTrianglesAroundEdge(i);
            if (shell.size() != 2)
                e2.is_activated = false;
            else
            {
                e2.is_activated = true;
                e2.m1 = -1;
                e2.m2 = -1;
                for (int j=0; j<3; j++)
                    if (_topology->getTriangle(shell[0]][j] != getEdge(i)[0] && _topology->getTriangle(shell[0])[j] != getEdge(i)[1])
                        e2.m1 = _topology->getTriangle(shell[0])[j];
                for (int j=0; j<3; j++)
                    if (_topology->getTriangle(shell[1])[j] != getEdge(i)[0] && _topology->getTriangle(shell[1])[j] != getEdge(i)[1])
                        e2.m2 = _topology->getTriangle(shell[1])[j];
                if (e2.m1 >= 0 && e2.m2 >= 0)
                {
                    e2.restlength = (x_rest[e2.m2]-x_rest[e2.m1]).norm();
                }
            }

            if (e2.is_activated != einfo->is_activated) serr << "ERROR: EdgeInfo["<<i<<"].is_activated = "<<einfo->is_activated<<" while it should be "<<e2.is_activated<<""<<sendl;
            else if (e2.is_activated)
            {
                if (!((e2.m1 == einfo->m1 && e2.m2 == einfo->m2) || (e2.m1 == einfo->m2 && e2.m2 == einfo->m1)))
                    serr << "ERROR: EdgeInfo["<<i<<"] points = "<<einfo->m1<<"-"<<einfo->m2<<" while it should be "<<e2.m1<<"-"<<e2.m2<<""<<sendl;
                if (e2.restlength != einfo->restlength)
                    serr << "ERROR: EdgeInfo["<<i<<"] length = "<<einfo->restlength<<" while it should be "<<e2.restlength<<""<<sendl;
            }
        }

#endif

        /*            serr<<"TriangularBendingSprings<DataTypes>::addForce() between "<<springs[i].m1<<" and "<<springs[i].m2<<sendl;*/

        if(einfo->is_activated)
        {
            //this->addSpringForce(m_potentialEnergy,f,x,v, i, einfo->spring);

            int a = einfo->m1;
            int b = einfo->m2;
            Coord u = x[b]-x[a];
            Real d = u.norm();
            if( d>1.0e-4 )
            {
                Real inverseLength = 1.0f/d;
                u *= inverseLength;
                Real elongation = (Real)(d - einfo->restlength);
                m_potentialEnergy += elongation * elongation * einfo->ks / 2;
                /*      serr<<"TriangularBendingSprings<DataTypes>::addSpringForce, p = "<<p<<sendl;

                		serr<<"TriangularBendingSprings<DataTypes>::addSpringForce, new potential energy = "<<potentialEnergy<<sendl;*/
                Deriv relativeVelocity = v[b]-v[a];
                Real elongationVelocity = dot(u,relativeVelocity);
                Real forceIntensity = (Real)(einfo->ks*elongation+einfo->kd*elongationVelocity);
                Deriv force = u*forceIntensity;
                f[a]+=force;
                f[b]-=force;

                updateMatrix=true;

                Mat& m = einfo->DfDx; //Mat& m = this->dfdx[i];
                Real tgt = forceIntensity * inverseLength;
                for( int j=0; j<N; ++j )
                {
                    for( int k=0; k<N; ++k )
                    {
                        m[j][k] = ((Real)einfo->ks-tgt) * u[j] * u[k];
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
    /*            serr<<"TriangularBendingSprings<DataTypes>::addForce() between "<<springs[i].m1<<" and "<<springs[i].m2<<sendl;*/
    //    this->addSpringForce(m_potentialEnergy,f,x,v, i, springs[i]);
    //}
}

template<class DataTypes>
void TriangularBendingSprings<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    int nbEdges=_topology->getNbEdges();
    const EdgeInformation *einfo;
    const helper::vector<EdgeInformation>& edgeInf = edgeInfo.getValue();

    df.resize(dx.size());
    //serr<<"TriangularBendingSprings<DataTypes>::addDForce, dx1 = "<<dx1<<sendl;
    //serr<<"TriangularBendingSprings<DataTypes>::addDForce, df1 before = "<<f1<<sendl;
    //const helper::vector<Spring>& springs = this->springs.getValue();

    for(int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInf[i];

        /*            serr<<"TriangularBendingSprings<DataTypes>::addForce() between "<<springs[i].m1<<" and "<<springs[i].m2<<sendl;*/

        if(einfo->is_activated)
        {
            //this->addSpringDForce(df,dx, i, einfo->spring);

            const int a = einfo->m1;
            const int b = einfo->m2;
            const Coord d = dx[b]-dx[a];
            const Deriv dforce = einfo->DfDx*d; //this->dfdx[i]*d;
            df[a]+= dforce * kFactor;
            df[b]-= dforce * kFactor;
            //serr<<"TriangularBendingSprings<DataTypes>::addSpringDForce, a="<<a<<", b="<<b<<", dforce ="<<dforce<<sendl;

            //if(updateMatrix){
            //}
            updateMatrix=false;
        }
    }
    d_df.endEdit();
    //for (unsigned int i=0; i<springs.size(); i++)
    //{
    //    this->addSpringDForce(df,dx, i, springs[i]);
    //}
    //serr<<"TriangularBendingSprings<DataTypes>::addDForce, df = "<<f<<sendl;
}


/*
template<class DataTypes>
void TriangularBendingSprings<DataTypes>::updateLameCoefficients()
{
	lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
	mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
//	serr << "initialized Lame coef : lambda=" <<lambda<< " mu="<<mu<<sendl;
}
*/


template<class DataTypes>
void TriangularBendingSprings<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    unsigned int i;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    //VecCoord& x_rest = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    //int nbTriangles=_topology->getNbTriangles();

    glDisable(GL_LIGHTING);


    /*
    glBegin(GL_TRIANGLES);
    for(i=0;i<nbTriangles; ++i)
    {
    	int a = _topology->getTriangle(i)[0];
    	int b = _topology->getTriangle(i)[1];
    	int c = _topology->getTriangle(i)[2];

    	glColor4f(0,1,0,1);
    	helper::gl::glVertexT(x[a]);
    	glColor4f(0,0.5,0.5,1);
    	helper::gl::glVertexT(x[b]);
    	glColor4f(0,0,1,1);
    	helper::gl::glVertexT(x[c]);
    }
    */
    unsigned int nb_to_draw = 0;

    const helper::vector<EdgeInformation>& edgeInf = edgeInfo.getValue();

    glBegin(GL_LINES);
    for(i=0; i<edgeInf.size(); ++i)
    {
        if(edgeInf[i].is_activated)
        {


            bool external=true;
            Real d = (x[edgeInf[i].m2]-x[edgeInf[i].m1]).norm();
            if (external)
            {
                if (d<edgeInf[i].restlength*0.9999)
                    glColor4f(1,0,0,1);
                else
                    glColor4f(0,1,0,1);
            }
            else
            {
                if (d<edgeInf[i].restlength*0.9999)
                    glColor4f(1,0.5f,0,1);
                else
                    glColor4f(0,1,0.5f,1);
            }


            nb_to_draw+=1;

            //glColor4f(0,1,0,1);
            helper::gl::glVertexT(x[edgeInf[i].m1]);
            helper::gl::glVertexT(x[edgeInf[i].m2]);

        }
    }
    glEnd();

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif /* SOFA_NO_OPENGL */
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif //#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARBENDINGSPRINGS_INL

