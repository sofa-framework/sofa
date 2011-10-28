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
#ifndef SOFA_COMPONENT_MAPPING_CATMULLROMSPLINEMAPPING_INL
#define SOFA_COMPONENT_MAPPING_CATMULLROMSPLINEMAPPING_INL

#include <sofa/component/mapping/CatmullRomSplineMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/core/Mapping.inl>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/gl/glText.inl>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/io/Mesh.h>
#include <limits>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/defaulttype/Vec.h>

#include <string>
#include <iostream>

namespace sofa
{

namespace component
{

namespace mapping
{

using helper::WriteAccessor;
using helper::ReadAccessor;
using sofa::defaulttype::Vec;

template <class TIn, class TOut>
CatmullRomSplineMapping<TIn, TOut>::CatmullRomSplineMapping ( )
    : Inherit ( )
    , SplittingLevel(initData(&SplittingLevel, (unsigned int) 0,"SplittingLevel","Number of recursive splits"))
    //, Radius(initData(&Radius,(Real)0.0,"Radius","Radius of the beam (generate triangle mesh if not null, a polyline otherwise)"))
{

}


template <class TIn, class TOut>
CatmullRomSplineMapping<TIn, TOut>::~CatmullRomSplineMapping ()
{
}



template <class TIn, class TOut>
void CatmullRomSplineMapping<TIn, TOut>::init()
{

    sourceMesh=this->getFromModel()->getContext()->getMeshTopology();
    targetMesh=this->getToModel()->getContext()->getMeshTopology() ;

    maskFrom = NULL;
    if ( core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *> ( this->getFromModel() ) )
        maskFrom = &stateFrom->forceMask;
    maskTo = NULL;
    if ( core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *> ( this->getToModel() ) )
        maskTo = &stateTo->forceMask;
    unsigned int k = SplittingLevel.getValue();
    unsigned int P = sourceMesh->getNbPoints();
    unsigned int E = sourceMesh->getNbEdges();
    SeqEdges Edges = sourceMesh->getEdges();

    // given the level of splitting and number of input edges E and points P , we can define the nb of mapped nodes
    // ( at each splitting, each edge is cut in two )
    // targetP =      P +  E ( 2^SplittingLevel -1 )
    // targetE =      E * 2^SplittingLevel

    unsigned int k2 = 1 << k;
    unsigned int targetP =  P + E * ( k2 - 1 ) ;
    unsigned int targetE =  E * k2 ;

    ReadAccessor<Data<VecInCoord> >		xfrom = *this->fromModel->read(core::ConstVecCoordId::restPosition());
    WriteAccessor<Data<VecOutCoord> >	xto0 = *this->toModel->write(core::VecCoordId::restPosition());
    WriteAccessor<Data<VecOutCoord> >	xto = *this->toModel->write(core::VecCoordId::position());
    WriteAccessor<Data<VecOutCoord> >	xtoReset = *this->toModel->write(core::VecCoordId::resetPosition());

    this->toModel->resize(targetP);
    targetMesh->setNbPoints(targetP);
    m_weight.resize(targetP);
    m_index.resize(targetP);

    unsigned int count=0;
    Vec<4,ID> id(0,0,0,0); vector<ID> n1,n2;
    OutReal t,t2,t3;
    for ( unsigned int i=0; i<P; i++ ) // initial points
    {
        xto0[count] = xfrom[i]; xto[count] = xto0[count]; xtoReset[count] = xto0[count];
        m_index[count][0] = m_index[count][1] = m_index[count][2] = m_index[count][3] = i;
        m_weight[count][0] = 1;
        m_weight[count][1] = m_weight[count][2] = m_weight[count][3] = 0;
        count++;
    }
    for ( unsigned int i=0; i<E; i++ ) // interpolated points
    {
        id[0]=id[1]=Edges[i][0];	n1=sourceMesh->getVerticesAroundVertex(id[1]);
        id[2]=id[3]=Edges[i][1];	n2=sourceMesh->getVerticesAroundVertex(id[2]);

        if(n1.size()>1) id[0]=(n1[0]==id[2])?n1[1]:n1[0];
        if(n2.size()>1) id[3]=(n2[0]==id[1])?n2[1]:n2[0];

        for ( unsigned int j=0; j<k2-1; j++ )
        {
            t=((OutReal)1.0+(OutReal)j)/(OutReal)k2; t2=t*t; t3=t2*t;
            xto0[count] = xfrom[id[1]]*t + xfrom[id[2]]*(1.0-t);
            xto[count] = xto0[count]; xtoReset[count] = xto0[count];
            m_index[count] = id;
            m_weight[count][0] = (-t3 + 2*t2 - t)/2.;
            m_weight[count][1] = (3*t3 - 5*t2 + 2)/2.;
            m_weight[count][2] = (-3*t3 + 4*t2 + t)/2.;
            m_weight[count][3] = (t3 - t2)/2.;

            count++;
        }
    }


    sofa::component::topology::EdgeSetTopologyModifier *to_estm;
    this->toModel->getContext()->get(to_estm);
    if(to_estm == NULL)
    {
        serr<<"EdgeSetTopologyModifier expected in the same context of CatmullRomSplineMapping"<<sendl;
    }
    else
    {
        vector< Edge >         edges_to_create  ; edges_to_create.resize (targetE);
        vector< unsigned int > edgesIndexList   ; edgesIndexList.resize  (targetE); for ( unsigned int i=0; i<targetE; i++ ) edgesIndexList[i]=i;
        count=0;
        for ( unsigned int i=0; i<E; i++ )
        {
            edges_to_create[count][0]=Edges[i][0];
            edges_to_create[count][1]=P+i*(k2-1);
            count++;
            for ( unsigned int j=1; j<k2-1; j++ )
            {
                edges_to_create[count][0]=edges_to_create[count-1][1];
                edges_to_create[count][1]=edges_to_create[count-1][1]+1;
                count++;
            }
            edges_to_create[count][0]=edges_to_create[count-1][1];
            edges_to_create[count][1]=Edges[i][1];
            count++;
        }

        to_estm->addEdgesProcess(edges_to_create) ;
        to_estm->addEdgesWarning(edges_to_create.size(), edges_to_create, edgesIndexList) ;
    }

    Inherit::init();
}


template <class TIn, class TOut>
void CatmullRomSplineMapping<TIn, TOut>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    for ( unsigned int i = 0 ; i < out.size(); i++ )
    {
        out[i] = OutCoord ();
        for ( unsigned int j=0; j<4 ; j++ )	out[i] += in[m_index[i][j]] * m_weight[i][j] ;
    }
}


template <class TIn, class TOut>
void CatmullRomSplineMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    if ( ! ( this->maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = OutDeriv();
            for ( unsigned int j=0; j<4 ; j++ )     out[i] += in[m_index[i][j]] * m_weight[i][j] ;
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;

        for ( it=indices.begin(); it!=indices.end(); it++ )
        {
            unsigned int i= ( unsigned int ) ( *it );
            out[i] = OutDeriv();
            for ( unsigned int j=0; j<4 ; j++ )  out[i] += in[m_index[i][j]] * m_weight[i][j] ;
        }
    }
}



template <class TIn, class TOut>
void CatmullRomSplineMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    if ( ! ( this->maskTo->isInUse() ) )
    {
        this->maskFrom->setInUse ( false );

        for ( unsigned int i=0; i<in.size(); i++ )
            for ( unsigned int j=0; j<4 ; j++ )
                out[m_index[i][j]]  += in[i] * m_weight[i][j];
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;

        for ( it=indices.begin(); it!=indices.end(); it++ )
        {
            const int i= ( int ) ( *it );
            for ( unsigned int j=0; j<4 ; j++ )
            {
                out[m_index[i][j]]  += in[i] * m_weight[i][j];
                maskFrom->insertEntry ( m_index[i][j] );
            }
        }
    }
}



template <class TIn, class TOut>
void CatmullRomSplineMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& parentJacobians, const typename Out::MatrixDeriv& childJacobians )
{
    for (typename Out::MatrixDeriv::RowConstIterator childJacobian = childJacobians.begin(); childJacobian != childJacobians.end(); ++childJacobian)
    {
        typename In::MatrixDeriv::RowIterator parentJacobian = parentJacobians.writeLine(childJacobian.index());

        for (typename Out::MatrixDeriv::ColConstIterator childParticle = childJacobian.begin(); childParticle != childJacobian.end(); ++childParticle)
        {
            unsigned int childIndex = childParticle.index();
            const OutDeriv& childJacobianVec = childParticle.val();

            for ( unsigned int j=0; j<4 ; j++ )
            {
                InDeriv parentJacobianVec;
                parentJacobianVec  += childJacobianVec * m_weight[childIndex][j];
                parentJacobian.addCol(m_index[childIndex][j],parentJacobianVec);
            }
        }
    }
}



template <class TIn, class TOut>
void CatmullRomSplineMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings()) return;

    const typename Out::VecCoord& xto = *this->toModel->getX();
    const typename In::VecCoord& xfrom = *this->fromModel->getX();

    glPushAttrib( GL_LIGHTING_BIT || GL_COLOR_BUFFER_BIT || GL_ENABLE_BIT);
    glDisable ( GL_LIGHTING );

    // Display mapping links between in and out elements
    glPointSize ( 1 );
    glBegin ( GL_LINES );

    for ( unsigned int i=0; i<xto.size(); i++ )
    {
        for ( unsigned int m=0 ; m<4 ; m++ )
        {
            if(m_weight[i][m]<0) glColor4d ( -m_weight[i][m]*4.0,0.5,0,1 );
            else glColor4d ( 0,0.5, m_weight[i][m],1 );
            helper::gl::glVertexT ( xfrom[m_index[i][m]] );
            helper::gl::glVertexT ( xto[i] );
        }
    }
    glEnd();

    glPopAttrib();
}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif

