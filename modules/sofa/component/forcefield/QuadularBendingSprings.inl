/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_QUADULARBENDINGSPRINGS_INL
#define SOFA_COMPONENT_FORCEFIELD_QUADULARBENDINGSPRINGS_INL

#include <sofa/component/forcefield/QuadularBendingSprings.h>
// #include <sofa/component/topology/MeshTopology.h>
#include <iostream>

#include <sofa/helper/gl/template.h>
#include <sofa/component/topology/QuadData.inl>
#include <sofa/component/topology/EdgeData.inl>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace	sofa::component::topology;
using namespace core::componentmodel::topology;

using std::cerr;
using std::cout;
using std::endl;

using namespace core::componentmodel::behavior;

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::QuadularBSEdgeCreationFunction(int /*edgeIndex*/, void* param, EdgeInformation &ei,
        const Edge& ,  const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    QuadularBendingSprings<DataTypes> *ff= (QuadularBendingSprings<DataTypes> *)param;
    if (ff)
    {
        //QuadSetTopology<DataTypes> *_mesh=ff->getQuadularTopology();
        //assert(_mesh!=0);
        unsigned int u,v;
        /// set to zero the edge stiffness matrix
        for (u=0; u<3; ++u)
        {
            for (v=0; v<3; ++v)
            {
                ei.DfDx[u][v]=0;
            }
        }

        ei.is_activated=false;
        ei.is_initialized=false;

    }
}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::QuadularBSQuadCreationFunction (const sofa::helper::vector<unsigned int> &quadAdded,
        void* param, vector<EdgeInformation> &edgeData)
{

    QuadularBendingSprings<DataTypes> *ff= (QuadularBendingSprings<DataTypes> *)param;
    if (ff)
    {
        QuadSetTopology<DataTypes> *_mesh=ff->getQuadularTopology();
        assert(_mesh!=0);
        QuadSetTopologyContainer *container=_mesh->getQuadSetTopologyContainer();

        //const sofa::helper::vector< Edge > &edgeArray=container->getEdgeArray() ;
        const sofa::helper::vector< Quad > &quadArray=container->getQuadArray() ;
        const sofa::helper::vector< QuadEdges > &quadEdgeArray=container->getQuadEdgeArray() ;

        double m_ks=ff->getKs();
        double m_kd=ff->getKd();

        unsigned int u,v;

        unsigned int nb_activated = 0;

        const typename DataTypes::VecCoord *restPosition=_mesh->getDOF()->getX0();

        for (unsigned int i=0; i<quadAdded.size(); ++i)
        {

            QuadSetTopology<DataTypes> *_mesh=ff->getQuadularTopology();
            assert(_mesh!=0);
            QuadSetTopologyContainer *container=_mesh->getQuadSetTopologyContainer();

            sofa::component::topology::QuadSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::QuadSetTopologyContainer *>(container);

            /// describe the jth edge index of quad no i
            QuadEdges te2 = quadEdgeArray[quadAdded[i]];
            /// describe the jth vertex index of quad no i
            Quad t2 = quadArray[quadAdded[i]];

            for(unsigned int j=0; j<4; ++j)
            {

                EdgeInformation &ei = edgeData[te2[j]]; // ff->edgeInfo
                if(!(ei.is_initialized))
                {

                    unsigned int edgeIndex = te2[j];
                    ei.is_activated=true;

                    /// set to zero the edge stiffness matrix
                    for (u=0; u<3; ++u)
                    {
                        for (v=0; v<3; ++v)
                        {
                            ei.DfDx[u][v]=0;
                        }
                    }

                    container->getQuadEdgeShellArray();
                    container->getQuadVertexShellArray();

                    const sofa::helper::vector< unsigned int > shell = tstc->getQuadEdgeShell(edgeIndex);
                    if (shell.size()==2)
                    {

                        nb_activated+=1;

                        QuadEdges te1;
                        Quad t1;

                        if(shell[0] == quadAdded[i])
                        {

                            te1 = quadEdgeArray[shell[1]];
                            t1 = quadArray[shell[1]];

                        }
                        else   // shell[1] == quadAdded[i]
                        {

                            te1 = quadEdgeArray[shell[0]];
                            t1 = quadArray[shell[0]];
                        }

                        int i1 = tstc->getEdgeIndexInQuad(te1, edgeIndex); //edgeIndex //te1[j]
                        int i2 = tstc->getEdgeIndexInQuad(te2, edgeIndex); // edgeIndex //te2[j]

                        ei.m1 = t1[i1]; // i1
                        ei.m2 = t2[(i2+3)%4]; // i2

                        ei.m3 = t1[(i1+3)%4]; // (i1+3)%4
                        ei.m4 = t2[i2]; // (i2+3)%4

                        //QuadularBendingSprings<DataTypes> *fftest= (QuadularBendingSprings<DataTypes> *)param;
                        ei.ks=m_ks; //(fftest->ks).getValue();
                        ei.kd=m_kd; //(fftest->kd).getValue();

                        Coord u1 = (*restPosition)[ei.m1] - (*restPosition)[ei.m2];
                        Real d1 = u1.norm();
                        ei.restlength1=(double) d1;

                        Coord u2 = (*restPosition)[ei.m3] - (*restPosition)[ei.m4];
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

    }

}

template< class DataTypes>
void QuadularBendingSprings<DataTypes>::QuadularBSQuadDestructionFunction (const sofa::helper::vector<unsigned int> &quadRemoved,
        void* param, vector<EdgeInformation> &edgeData)
{
    QuadularBendingSprings<DataTypes> *ff= (QuadularBendingSprings<DataTypes> *)param;
    if (ff)
    {
        QuadSetTopology<DataTypes> *_mesh=ff->getQuadularTopology();
        assert(_mesh!=0);
        QuadSetTopologyContainer *container=_mesh->getQuadSetTopologyContainer();
        //const sofa::helper::vector< Edge > &edgeArray=container->getEdgeArray() ;
        const sofa::helper::vector< Quad > &quadArray=container->getQuadArray() ;
        const sofa::helper::vector< QuadEdges > &quadEdgeArray=container->getQuadEdgeArray() ;

        double m_ks=ff->getKs(); // typename DataTypes::
        double m_kd=ff->getKd(); // typename DataTypes::

        //unsigned int u,v;

        const typename DataTypes::VecCoord *restPosition=_mesh->getDOF()->getX0();

        for (unsigned int i=0; i<quadRemoved.size(); ++i)
        {

            QuadSetTopology<DataTypes> *_mesh=ff->getQuadularTopology();
            assert(_mesh!=0);
            QuadSetTopologyContainer *container=_mesh->getQuadSetTopologyContainer();

            sofa::component::topology::QuadSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::QuadSetTopologyContainer *>(container);

            /// describe the jth edge index of quad no i
            QuadEdges te = quadEdgeArray[quadRemoved[i]];
            /// describe the jth vertex index of quad no i
            Quad t = quadArray[quadRemoved[i]];


            for(unsigned int j=0; j<4; ++j)
            {

                EdgeInformation &ei = edgeData[te[j]]; // ff->edgeInfo
                if(ei.is_initialized)
                {

                    unsigned int edgeIndex = te[j];

                    container->getQuadEdgeShellArray();
                    container->getQuadVertexShellArray();

                    const sofa::helper::vector< unsigned int > shell = tstc->getQuadEdgeShell(edgeIndex);
                    if (shell.size()==3)
                    {

                        QuadEdges te1;
                        Quad t1;
                        QuadEdges te2;
                        Quad t2;

                        if(shell[0] == quadRemoved[i])
                        {
                            te1 = quadEdgeArray[shell[1]];
                            t1 = quadArray[shell[1]];
                            te2 = quadEdgeArray[shell[2]];
                            t2 = quadArray[shell[2]];

                        }
                        else
                        {

                            if(shell[1] == quadRemoved[i])
                            {

                                te1 = quadEdgeArray[shell[2]];
                                t1 = quadArray[shell[2]];
                                te2 = quadEdgeArray[shell[0]];
                                t2 = quadArray[shell[0]];

                            }
                            else   // shell[2] == quadRemoved[i]
                            {

                                te1 = quadEdgeArray[shell[0]];
                                t1 = quadArray[shell[0]];
                                te2 = quadEdgeArray[shell[1]];
                                t2 = quadArray[shell[1]];

                            }
                        }

                        int i1 = tstc->getEdgeIndexInQuad(te1, edgeIndex);
                        int i2 = tstc->getEdgeIndexInQuad(te2, edgeIndex);

                        ei.m1 = t1[i1];
                        ei.m2 = t2[(i2+3)%4];

                        ei.m3 = t1[(i1+3)%4];
                        ei.m4 = t2[i2];

                        //QuadularBendingSprings<DataTypes> *fftest= (QuadularBendingSprings<DataTypes> *)param;
                        ei.ks=m_ks; //(fftest->ks).getValue();
                        ei.kd=m_kd; //(fftest->kd).getValue();

                        Coord u1 = (*restPosition)[ei.m1] - (*restPosition)[ei.m2];
                        Real d1 = u1.norm();
                        ei.restlength1=(double) d1;

                        Coord u2 = (*restPosition)[ei.m3] - (*restPosition)[ei.m4];
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

    }
}


template<class DataTypes>
QuadularBendingSprings<DataTypes>::QuadularBendingSprings()
    : _mesh(NULL)
    , updateMatrix(true)
    , f_ks(initData(&f_ks,(double) 100000.0,"stiffness","uniform stiffness for the all springs")) //(Real)0.3 ??
    , f_kd(initData(&f_kd,(double) 1.0,"damping","uniform damping for the all springs")) // (Real)1000. ??
{
}


template<class DataTypes>
QuadularBendingSprings<DataTypes>::~QuadularBendingSprings()
{}


template <class DataTypes> void QuadularBendingSprings<DataTypes>::handleTopologyChange()
{
    bool debug_mode = false;

    sofa::core::componentmodel::topology::BaseTopology *topology = static_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());

    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    edgeInfo.handleTopologyEvents(itBegin,itEnd);
    //quadInfo.handleTopologyEvents(itBegin,itEnd);

    while( itBegin != itEnd )
    {
        core::componentmodel::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();

        sofa::core::componentmodel::topology::BaseTopology* bt = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());
        sofa::core::componentmodel::topology::TopologyContainer *container=bt->getTopologyContainer();

        sofa::component::topology::QuadSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::QuadSetTopologyContainer *>(container);
        const sofa::helper::vector< Quad > &quadArray=tstc->getQuadArray() ;
        const sofa::helper::vector< QuadEdges > &quadEdgeArray=tstc->getQuadEdgeArray() ;


        if(tstc && changeType == core::componentmodel::topology::POINTSREMOVED)
        {


            const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=tstc->getQuadVertexShellArray();
            unsigned int last = tvsa.size() -1;
            unsigned int i,j;

            const sofa::helper::vector<unsigned int> tab = ( dynamic_cast< const sofa::component::topology::PointsRemoved * >( *itBegin ) )->getArray();

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

                const sofa::helper::vector<unsigned int> &shell= tvsa[lastIndexVec[i]];
                for (j=0; j<shell.size(); ++j)
                {

                    Quad tj = quadArray[shell[j]];

                    unsigned int vertexIndex = tstc->getVertexIndexInQuad(tj, lastIndexVec[i]);

                    QuadEdges tej = quadEdgeArray[shell[j]];

                    for (unsigned int j_edge=vertexIndex; j_edge%4 !=(vertexIndex+2)%4; ++j_edge)
                    {

                        unsigned int ind_j = tej[j_edge%4];

                        if (edgeInfo[ind_j].m1 == (int) last)
                        {
                            edgeInfo[ind_j].m1=(int) tab[i];
                            //std::cout << "INFO_print : OK m1 for ind_j =" << ind_j << std::endl;
                        }
                        else
                        {
                            if (edgeInfo[ind_j].m2 == (int) last)
                            {
                                edgeInfo[ind_j].m2=(int) tab[i];
                                //std::cout << "INFO_print : OK m2 for ind_j =" << ind_j << std::endl;
                            }
                        }

                        if (edgeInfo[ind_j].m3 == (int) last)
                        {
                            edgeInfo[ind_j].m3=(int) tab[i];
                            //std::cout << "INFO_print : OK m3 for ind_j =" << ind_j << std::endl;
                        }
                        else
                        {
                            if (edgeInfo[ind_j].m4 == (int) last)
                            {
                                edgeInfo[ind_j].m4=(int) tab[i];
                                //std::cout << "INFO_print : OK m4 for ind_j =" << ind_j << std::endl;
                            }
                        }

                    }
                }

                if(debug_mode)
                {

                    for (unsigned int j_loc=0; j_loc<edgeInfo.size(); ++j_loc)
                    {

                        bool is_forgotten = false;
                        if (edgeInfo[j_loc].m1 == (int) last)
                        {
                            edgeInfo[j_loc].m1 =(int) tab[i];
                            is_forgotten=true;
                            //std::cout << "INFO_print : QuadularBendingSprings - MISS m1 for j_loc =" << j_loc << std::endl;

                        }
                        else
                        {
                            if (edgeInfo[j_loc].m2 ==(int) last)
                            {
                                edgeInfo[j_loc].m2 =(int) tab[i];
                                is_forgotten=true;
                                //std::cout << "INFO_print : QuadularBendingSprings - MISS m2 for j_loc =" << j_loc << std::endl;

                            }

                        }

                        if (edgeInfo[j_loc].m3 == (int) last)
                        {
                            edgeInfo[j_loc].m3 =(int) tab[i];
                            is_forgotten=true;
                            //std::cout << "INFO_print : QuadularBendingSprings - MISS m3 for j_loc =" << j_loc << std::endl;

                        }
                        else
                        {
                            if (edgeInfo[j_loc].m4 ==(int) last)
                            {
                                edgeInfo[j_loc].m4 =(int) tab[i];
                                is_forgotten=true;
                                //std::cout << "INFO_print : QuadularBendingSprings - MISS m4 for j_loc =" << j_loc << std::endl;

                            }

                        }
                    }
                }

                --last;
            }

        }
        else
        {
            if(tstc && changeType == core::componentmodel::topology::POINTSRENUMBERING)
            {

                sofa::component::topology::EdgeSetTopologyContainer *estc= dynamic_cast<sofa::component::topology::EdgeSetTopologyContainer *>(container);
                const sofa::helper::vector<sofa::component::topology::Edge> &ea=estc->getEdgeArray();

                unsigned int i;

                const sofa::helper::vector<unsigned int> tab = ( dynamic_cast< const sofa::component::topology::PointsRenumbering * >( *itBegin ) )->getinv_IndexArray();

                for ( i = 0; i < ea.size(); ++i)
                {
                    if(edgeInfo[i].is_activated)
                    {
                        edgeInfo[i].m1  = tab[edgeInfo[i].m1];
                        edgeInfo[i].m2  = tab[edgeInfo[i].m2];
                        edgeInfo[i].m3  = tab[edgeInfo[i].m3];
                        edgeInfo[i].m4  = tab[edgeInfo[i].m4];
                    }
                }
            }
        }
        ++itBegin;
    }
}


template<class DataTypes>
void QuadularBendingSprings<DataTypes>::init()
{
    //std::cerr << "initializing QuadularBendingSprings" << std::endl;
    this->Inherited::init();

    _mesh =0;
    if (getContext()->getMainTopology()!=0)
        _mesh= dynamic_cast<QuadSetTopology<DataTypes>*>(getContext()->getMainTopology());

    if ((_mesh==0) || (_mesh->getQuadSetTopologyContainer()->getNumberOfQuads()==0))
    {
        std::cerr << "ERROR(QuadularBendingSprings): object must have a Quadular Set Topology.\n";
        return;
    }

    QuadSetTopologyContainer *container=_mesh->getQuadSetTopologyContainer();

    /// prepare to store info in the edge array
    edgeInfo.resize(container->getNumberOfEdges());

    unsigned int i;
    // set edge tensor to 0
    const sofa::helper::vector<Edge> &edgeArray=container->getEdgeArray();
    for (i=0; i<container->getNumberOfEdges(); ++i)
    {

        QuadularBSEdgeCreationFunction(i, (void*) this, edgeInfo[i],
                edgeArray[i],  (const sofa::helper::vector< unsigned int > )0,
                (const sofa::helper::vector< double >)0);
    }

    // create edge tensor by calling the quad creation function
    sofa::helper::vector<unsigned int> quadAdded;
    for (i=0; i<container->getNumberOfQuads(); ++i)
    {
        quadAdded.push_back(i);
    }
    QuadularBSQuadCreationFunction(quadAdded,(void*) this,
            edgeInfo);

    edgeInfo.setCreateFunction(QuadularBSEdgeCreationFunction);
    edgeInfo.setCreateQuadFunction(QuadularBSQuadCreationFunction);
    edgeInfo.setDestroyQuadFunction(QuadularBSQuadDestructionFunction);
    edgeInfo.setCreateParameter( (void *) this );
    edgeInfo.setDestroyParameter( (void *) this );
    /////

    /*
    dof = dynamic_cast<MechanicalObject<DataTypes>*>( this->getContext()->getMechanicalState() );
    assert(dof);
    //std::cout<<"==================================QuadularBendingSprings<DataTypes>::init(), dof size = "<<dof->getX()->size()<<std::endl;

    // Set the bending springs

    std::map< IndexPair, IndexPair > edgeMap;
    std::set< IndexPair > springSet;

    topology::MeshTopology* topology = dynamic_cast<topology::MeshTopology*>( this->getContext()->getTopology() );
    assert( topology );

    const topology::MeshTopology::SeqQuads& quads = topology->getQuads();
    //std::cout<<"==================================QuadularBendingSprings<DataTypes>::init(), quads size = "<<quads.size()<<std::endl;
    for( unsigned i= 0; i<quads.size(); ++i )
    {
        const topology::MeshTopology::Quad& face = quads[i];
        {
            registerEdge( std::make_pair(face[0], face[1]), std::make_pair(face[3], face[2]), edgeMap, springSet );
            registerEdge( std::make_pair(face[1], face[2]), std::make_pair(face[0], face[3]), edgeMap, springSet );
            registerEdge( std::make_pair(face[2], face[3]), std::make_pair(face[1], face[0]), edgeMap, springSet );
            registerEdge( std::make_pair(face[3], face[0]), std::make_pair(face[2], face[1]), edgeMap, springSet );
        }
    }

    // init the parent class
    StiffSpringForceField<DataTypes>::init();
    */
}

template <class DataTypes>
sofa::defaulttype::Vector3::value_type QuadularBendingSprings<DataTypes>::getPotentialEnergy(const VecCoord& /*x*/)
{
    std::cerr<<"QuadularBendingSprings::getPotentialEnergy-not-implemented !!!"<<endl;
    return 0;
}

template<class DataTypes>
void QuadularBendingSprings<DataTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{

    QuadSetTopologyContainer *container=_mesh->getQuadSetTopologyContainer();
    unsigned int nbEdges=container->getNumberOfEdges();
    //const sofa::helper::vector<Edge> &edgeArray=container->getEdgeArray();

    EdgeInformation *einfo;

    //const helper::vector<Spring>& m_springs= this->springs.getValue();
    //this->dfdx.resize(nbEdges); //m_springs.size()
    f.resize(x.size());
    m_potentialEnergy = 0;
    /*        cerr<<"QuadularBendingSprings<DataTypes>::addForce()"<<endl;*/


//		sofa::core::componentmodel::topology::BaseTopology* bt = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());
//		sofa::core::componentmodel::topology::TopologyContainer *container=bt->getTopologyContainer();

//		sofa::component::topology::QuadSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::QuadSetTopologyContainer *>(container);


    for(unsigned int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInfo[i];

        /*            cerr<<"QuadularBendingSprings<DataTypes>::addForce() between "<<springs[i].m1<<" and "<<springs[i].m2<<endl;*/

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
                /*      cerr<<"QuadularBendingSprings<DataTypes>::addSpringForce, p = "<<p<<endl;

                		cerr<<"QuadularBendingSprings<DataTypes>::addSpringForce, new potential energy = "<<potentialEnergy<<endl;*/
                Deriv relativeVelocity = v[b1]-v[a1];
                Real elongationVelocity = dot(u1,relativeVelocity);
                Real forceIntensity = (Real)(einfo->ks*elongation+einfo->kd*elongationVelocity);
                Deriv force = u1*forceIntensity;
                f[a1]+=force;
                f[b1]-=force;

                updateMatrix=true;

                Mat3& m = einfo->DfDx; //Mat& m = this->dfdx[i];
                Real tgt = forceIntensity * inverseLength;
                for( int j=0; j<3; ++j )
                {
                    for( int k=0; k<3; ++k )
                    {
                        m[j][k] = ((Real)einfo->ks-tgt) * u1[j] * u1[k];
                    }
                    m[j][j] += tgt;
                }
            }
            else // null length, no force and no stiffness
            {
                Mat3& m = einfo->DfDx; //Mat& m = this->dfdx[i];
                for( int j=0; j<3; ++j )
                {
                    for( int k=0; k<3; ++k )
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
                /*      cerr<<"QuadularBendingSprings<DataTypes>::addSpringForce, p = "<<p<<endl;

                		cerr<<"QuadularBendingSprings<DataTypes>::addSpringForce, new potential energy = "<<potentialEnergy<<endl;*/
                Deriv relativeVelocity = v[b2]-v[a2];
                Real elongationVelocity = dot(u2,relativeVelocity);
                Real forceIntensity = (Real)(einfo->ks*elongation+einfo->kd*elongationVelocity);
                Deriv force = u2*forceIntensity;
                f[a2]+=force;
                f[b2]-=force;

                updateMatrix=true;

                Mat3& m = einfo->DfDx; //Mat& m = this->dfdx[i];
                Real tgt = forceIntensity * inverseLength;
                for( int j=0; j<3; ++j )
                {
                    for( int k=0; k<3; ++k )
                    {
                        m[j][k] = ((Real)einfo->ks-tgt) * u2[j] * u2[k];
                    }
                    m[j][j] += tgt;
                }
            }
            else // null length, no force and no stiffness
            {
                Mat3& m = einfo->DfDx; //Mat& m = this->dfdx[i];
                for( int j=0; j<3; ++j )
                {
                    for( int k=0; k<3; ++k )
                    {
                        m[j][k] = 0;
                    }
                }
            }

        }
    }


    //for (unsigned int i=0; i<springs.size(); i++)
    //{
    /*            cerr<<"QuadularBendingSprings<DataTypes>::addForce() between "<<springs[i].m1<<" and "<<springs[i].m2<<endl;*/
    //    this->addSpringForce(m_potentialEnergy,f,x,v, i, springs[i]);
    //}
}

template<class DataTypes>
void QuadularBendingSprings<DataTypes>::addDForce(VecDeriv& df, const VecDeriv& dx)
{

    QuadSetTopologyContainer *container=_mesh->getQuadSetTopologyContainer();
    unsigned int nbEdges=container->getNumberOfEdges();
    //const sofa::helper::vector<Edge> &edgeArray=container->getEdgeArray();

    EdgeInformation *einfo;

    df.resize(dx.size());
    //cerr<<"QuadularBendingSprings<DataTypes>::addDForce, dx1 = "<<dx1<<endl;
    //cerr<<"QuadularBendingSprings<DataTypes>::addDForce, df1 before = "<<f1<<endl;
    //const helper::vector<Spring>& springs = this->springs.getValue();

    for(unsigned int i=0; i<nbEdges; i++ )
    {
        einfo=&edgeInfo[i];

        /*            cerr<<"QuadularBendingSprings<DataTypes>::addForce() between "<<springs[i].m1<<" and "<<springs[i].m2<<endl;*/

        if(einfo->is_activated)
        {
            //this->addSpringDForce(df,dx, i, einfo->spring);

            const int a1 = einfo->m1;
            const int b1 = einfo->m2;
            const Coord d1 = dx[b1]-dx[a1];
            const int a2 = einfo->m3;
            const int b2 = einfo->m4;
            const Coord d2 = dx[b2]-dx[a2];
            const Deriv dforce1 = einfo->DfDx*d1;
            const Deriv dforce2 = einfo->DfDx*d2;
            df[a1]+=dforce1;
            df[b1]-=dforce1;
            df[a2]+=dforce2;
            df[b2]-=dforce2;
            //cerr<<"QuadularBendingSprings<DataTypes>::addSpringDForce, a="<<a<<", b="<<b<<", dforce ="<<dforce<<endl;

            if(updateMatrix)
            {

            }
            updateMatrix=false;
        }
    }

    //for (unsigned int i=0; i<springs.size(); i++)
    //{
    //    this->addSpringDForce(df,dx, i, springs[i]);
    //}
    //cerr<<"QuadularBendingSprings<DataTypes>::addDForce, df = "<<f<<endl;
}


/*
template<class DataTypes>
void QuadularBendingSprings<DataTypes>::updateLameCoefficients()
{
	lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
	mu = f_youngModulus.getValue()*(1-f_poissonRatio.getValue())/(1-f_poissonRatio.getValue()*f_poissonRatio.getValue());
//	std::cerr << "initialized Lame coef : lambda=" <<lambda<< " mu="<<mu<<std::endl;
}
*/


template<class DataTypes>
void QuadularBendingSprings<DataTypes>::draw()
{
    unsigned int i;
    if (!getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    VecCoord& x = *this->mstate->getX();

    glDisable(GL_LIGHTING);

    unsigned int nb_to_draw = 0;

    glBegin(GL_LINES);
    for(i=0; i<edgeInfo.size(); ++i)
    {
        if(edgeInfo[i].is_activated)
        {


            bool external=true;
            Real d1 = (x[edgeInfo[i].m2]-x[edgeInfo[i].m1]).norm();
            if (external)
            {
                if (d1<edgeInfo[i].restlength1*0.9999)
                    glColor4f(1,0,0,1);
                else
                    glColor4f(0,1,0,1);
            }
            else
            {
                if (d1<edgeInfo[i].restlength1*0.9999)
                    glColor4f(1,0.5f,0,1);
                else
                    glColor4f(0,1,0.5f,1);
            }


            nb_to_draw+=1;

            //glColor4f(0,1,0,1);
            helper::gl::glVertexT(x[edgeInfo[i].m1]);
            helper::gl::glVertexT(x[edgeInfo[i].m2]);

            Real d2 = (x[edgeInfo[i].m4]-x[edgeInfo[i].m3]).norm();
            if (external)
            {
                if (d2<edgeInfo[i].restlength2*0.9999)
                    glColor4f(1,0,0,1);
                else
                    glColor4f(0,1,0,1);
            }
            else
            {
                if (d2<edgeInfo[i].restlength2*0.9999)
                    glColor4f(1,0.5f,0,1);
                else
                    glColor4f(0,1,0.5f,1);
            }


            nb_to_draw+=1;

            //glColor4f(0,1,0,1);
            helper::gl::glVertexT(x[edgeInfo[i].m3]);
            helper::gl::glVertexT(x[edgeInfo[i].m4]);

        }
    }
    glEnd();


    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    ////

    sofa::core::componentmodel::topology::BaseTopology* bt = dynamic_cast<sofa::core::componentmodel::topology::BaseTopology *>(getContext()->getMainTopology());
    sofa::core::componentmodel::topology::TopologyContainer *container=bt->getTopologyContainer();

    sofa::component::topology::QuadSetTopologyContainer *tstc= dynamic_cast<sofa::component::topology::QuadSetTopologyContainer *>(container);
    const sofa::helper::vector< Quad > &quadArray=tstc->getQuadArray() ;

    glBegin(GL_QUADS);
    glColor4f(1,0,0,1);
    for(i=0; i<quadArray.size(); ++i)
    {
        helper::gl::glVertexT(x[quadArray[i][0]]);
        helper::gl::glVertexT(x[quadArray[i][1]]);
        helper::gl::glVertexT(x[quadArray[i][2]]);
        helper::gl::glVertexT(x[quadArray[i][3]]);
    }
    glEnd();

    ////
}



} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
