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
#ifndef SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_INL
#define SOFA_COMPONENT_MAPPING_SKINNINGMAPPING_INL

#include <sofa/component/mapping/SkinningMapping.h>
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
SkinningMapping<TIn, TOut>::SkinningMapping (core::State<In>* from, core::State<Out>* to )
    : Inherit ( from, to )
    , f_initPos ( initData ( &f_initPos,"initPos","initial child coordinates in the world reference frame." ) )
    , nbRef ( initData ( &nbRef, ( unsigned ) 4,"nbRef","Number of primitives influencing each point." ) )
    , f_index ( initData ( &f_index,"indices","parent indices for each child." ) )
    , weight ( initData ( &weight,"weight","influence weights of the Dofs." ) )
    , showFromIndex ( initData ( &showFromIndex, ( unsigned int ) 0, "showFromIndex","Displayed From Index." ) )
    , showWeights ( initData ( &showWeights, false, "showWeights","Show influence." ) )
{
    maskFrom = NULL;
    if ( core::behavior::BaseMechanicalState *stateFrom = dynamic_cast< core::behavior::BaseMechanicalState *> ( from ) )
        maskFrom = &stateFrom->forceMask;
    maskTo = NULL;
    if ( core::behavior::BaseMechanicalState *stateTo = dynamic_cast< core::behavior::BaseMechanicalState *> ( to ) )
        maskTo = &stateTo->forceMask;
}


template <class TIn, class TOut>
SkinningMapping<TIn, TOut>::~SkinningMapping ()
{
}



template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::init()
{
    unsigned int numChildren = this->toModel->getSize();
    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    WriteAccessor<Data<VecOutCoord> > initPos(this->f_initPos);

    if( this->f_initPos.getValue().size() != numChildren )
    {
        initPos.resize(out.size());
        for(unsigned int i=0; i<out.size(); i++ )
            initPos[i] = out[i];
    }

    if( this->weight.getValue().size() != numChildren || this->f_index.getValue().size() != numChildren)
        updateWeights(); // if not defined by user -> recompute based on euclidean distances

    reinit();
    Inherit::init();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::reinit()
{
    unsigned int nbref=nbRef.getValue();

    ReadAccessor<Data<VecOutCoord> > out (*this->toModel->read(core::ConstVecCoordId::position()));
    ReadAccessor<Data<VecOutCoord> > xto (this->f_initPos);
    ReadAccessor<Data<VecInCoord> > xfrom = *this->fromModel->read(core::ConstVecCoordId::restPosition());
    WriteAccessor<Data<vector<SVector<InReal> > > > m_weights  ( weight );
    ReadAccessor<Data<vector<SVector<unsigned int> > > > index ( this->f_index );

    // normalize weights
    for (unsigned int i=0; i<xto.size(); i++ )
    {
        InReal w=0;  for (unsigned int j=0; j<nbref; j++ ) w+=m_weights[i][j];
        if(w!=0) for (unsigned int j=0; j<nbref; j++ ) m_weights[i][j]/=w;
    }

    // precompute local/rotated positions

    f_localPos.resize(out.size());
    f_rotatedPos.resize(out.size());

    for(unsigned int i=0; i<out.size(); i++ )
    {
        Vec<3,InReal> cto; Out::get( cto[0],cto[1],cto[2], xto[i] );
        f_localPos[i].resize(nbref);
        f_rotatedPos[i].resize(nbref);

        for (unsigned int j=0 ; j<nbref; j++ )
        {
            f_localPos[i][j]= xfrom[index[i][j]].pointToChild(cto) * m_weights[i][j];
            f_rotatedPos[i][j]= (cto - xfrom[index[i][j]].getCenter() ) * m_weights[i][j];
        }
    }

}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::updateWeights ()
{
    ReadAccessor<Data<VecOutCoord> > xto (this->f_initPos);
    ReadAccessor<Data<VecInCoord> > xfrom = *this->fromModel->read(core::ConstVecCoordId::restPosition());
    WriteAccessor<Data<vector<SVector<InReal> > > > m_weights  ( weight );
    WriteAccessor<Data<vector<SVector<unsigned int> > > > index ( f_index );

    index.resize( xto.size() );
    m_weights.resize ( xto.size() );
    unsigned int nbref=nbRef.getValue();

    // compute 1/d^2 weights with Euclidean distance
    for (unsigned int i=0; i<xto.size(); i++ )
    {
        Vec<3,InReal> cto; Out::get( cto[0],cto[1],cto[2], xto[i] );

        // get the nbRef closest primitives
        index[i].resize( nbref );
        m_weights[i].resize ( nbref );
        for (unsigned int j=0; j<nbref; j++ )
        {
            m_weights[i][j]=0;
            index[i][j]=0;
        }
        for (unsigned int j=0; j<xfrom.size(); j++ )
        {
            Vec<3,InReal> cfrom; In::get( cfrom[0],cfrom[1],cfrom[2], xfrom[j] );
            InReal w=(cto-cfrom)*(cto-cfrom);
            if(w!=0) w=1./w;
            else w=std::numeric_limits<InReal>::max();
            unsigned int m=0; while (m!=nbref && m_weights[i][m]>w) m++;
            if(m!=nbref)
            {
                for (unsigned int k=nbref-1; k>m; k--)
                {
                    m_weights[i][k]=m_weights[i][k-1];
                    index[i][k]=index[i][k-1];
                }
                m_weights[i][m]=w;
                index[i][m]=j;
            }
        }

        // normalization
        InReal w=0;  for (unsigned int j=0; j<nbref; j++ ) w+=m_weights[i][j];
        if(w!=0) for (unsigned int j=0; j<nbref; j++ ) m_weights[i][j]/=w;
    }

}



template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    unsigned int nbref=nbRef.getValue();
    ReadAccessor<Data<vector<SVector<InReal> > > > m_weights  ( this->weight );
    ReadAccessor<Data<vector<SVector<unsigned int> > > > index ( f_index );

    for ( unsigned int i = 0 ; i < out.size(); i++ )
    {
        out[i] = OutCoord ();
        for ( unsigned int j=0; j<nbref && m_weights[i][j]>0.; j++ )
        {
            f_rotatedPos[i][j]=in[index[i][j]].rotate(f_localPos[i][j]);
            out[i] += in[index[i][j]].getCenter() * m_weights[i][j] + f_rotatedPos[i][j];
        }
    }
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    unsigned int nbref=nbRef.getValue();
    ReadAccessor<Data<vector<SVector<InReal> > > > m_weights  ( weight );
    ReadAccessor<Data<vector<SVector<unsigned int> > > > index ( f_index );

    if ( ! ( this->maskTo->isInUse() ) )
    {
        for ( unsigned int i=0; i<out.size(); i++ )
        {
            out[i] = OutDeriv();
            for ( unsigned int j=0; j<nbref && m_weights[i][j]>0.; j++ )
            {
                out[i] += getLinear( in[index[i][j]] ) * m_weights[i][j] + cross(getAngular(in[index[i][j]]), f_rotatedPos[i][j]);
            }
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
            for ( unsigned int j=0; j<nbref && m_weights[i][j]>0.; j++ )
            {
                out[i] += getLinear( in[index[i][j]] ) * m_weights[i][j] + cross(getAngular(in[index[i][j]]), f_rotatedPos[i][j]);
            }
        }
    }
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    unsigned int nbref=nbRef.getValue();
    ReadAccessor<Data<vector<SVector<InReal> > > > m_weights  ( weight );
    ReadAccessor<Data<vector<SVector<unsigned int> > > > index ( f_index );

    if ( ! ( this->maskTo->isInUse() ) )
    {
        this->maskFrom->setInUse ( false );

        for ( unsigned int i=0; i<in.size(); i++ )
        {
            for ( unsigned int j=0; j<nbref && m_weights[i][j]>0.; j++ )
            {
                getLinear(out[index[i][j]])  += in[i] * m_weights[i][j];
                getAngular(out[index[i][j]]) += cross(f_rotatedPos[i][j], in[i]);
            }
        }
    }
    else
    {
        typedef helper::ParticleMask ParticleMask;
        const ParticleMask::InternalStorage &indices=this->maskTo->getEntries();

        ParticleMask::InternalStorage::const_iterator it;

        for ( it=indices.begin(); it!=indices.end(); it++ )
        {
            const int i= ( int ) ( *it );

            for ( unsigned int j=0; j<nbref && m_weights[i][j]>0.; j++ )
            {
                getLinear(out[index[i][j]])  += in[i] * m_weights[i][j];
                getAngular(out[index[i][j]]) += cross(f_rotatedPos[i][j], in[i]);
                maskFrom->insertEntry ( index[i][j] );
            }
        }
    }
}



template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJT ( typename In::MatrixDeriv& parentJacobians, const typename Out::MatrixDeriv& childJacobians )
{
    unsigned int nbref=nbRef.getValue();
    ReadAccessor<Data<vector<SVector<InReal> > > > m_weights  ( weight );
    ReadAccessor<Data<vector<SVector<unsigned int> > > > index ( f_index );

    for (typename Out::MatrixDeriv::RowConstIterator childJacobian = childJacobians.begin(); childJacobian != childJacobians.end(); ++childJacobian)
    {
        typename In::MatrixDeriv::RowIterator parentJacobian = parentJacobians.writeLine(childJacobian.index());

        for (typename Out::MatrixDeriv::ColConstIterator childParticle = childJacobian.begin(); childParticle != childJacobian.end(); ++childParticle)
        {
            unsigned int childIndex = childParticle.index();
            const OutDeriv& childJacobianVec = childParticle.val();

            for ( unsigned int j=0; j<nbref && m_weights[childIndex][j]>0.; j++ )
            {
                InDeriv parentJacobianVec;
                getLinear(parentJacobianVec)  += childJacobianVec * m_weights[childIndex][j];
                getAngular(parentJacobianVec) += cross(f_rotatedPos[childIndex][j], childJacobianVec);
                parentJacobian.addCol(index[childIndex][j],parentJacobianVec);
            }
        }
    }
}



template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    const typename Out::VecCoord& xto = *this->toModel->getX();
    const typename In::VecCoord& xfrom = *this->fromModel->getX();
    const unsigned int nbref = this->nbRef.getValue();

    ReadAccessor<Data<vector<SVector<InReal> > > > m_weights  ( weight );
    ReadAccessor<Data<vector<SVector<unsigned int> > > > index ( f_index );

    glPushAttrib( GL_LIGHTING_BIT || GL_COLOR_BUFFER_BIT || GL_ENABLE_BIT);
    glDisable ( GL_LIGHTING );

    if ( vparams->displayFlags().getShowMappings() )
    {
        // Display mapping links between in and out elements
        glPointSize ( 1 );
        glColor4f ( 1,1,0,1 );
        glBegin ( GL_LINES );

        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            for ( unsigned int m=0 ; m<nbref && m_weights[i][m]>0.; m++ )
            {
                glColor4d ( m_weights[i][m],m_weights[i][m],0,1 );
                helper::gl::glVertexT ( xfrom[index[i][m]].getCenter() );
                helper::gl::glVertexT ( xto[i] );
            }
        }
        glEnd();
    }

    // Show weights
    if ( showWeights.getValue())
    {
        InReal minValue = 1, maxValue = 0;


        if ( ! triangles.empty()) // Show on mesh
        {
            std::vector< defaulttype::Vector3 > points;
            std::vector< defaulttype::Vector3 > normals;
            std::vector< defaulttype::Vec<4,float> > colors;
            for ( unsigned int i = 0; i < triangles.size(); i++)
            {
                for ( unsigned int j = 0; j < 3; j++)
                {
                    const unsigned int& indexPoint = triangles[i][j];
                    double color = 0;

                    for ( unsigned int m=0 ; m<nbref && m_weights[indexPoint][m]>0.; m++ )
                        if(index[indexPoint][m]==showFromIndex.getValue())
                            color = (m_weights[indexPoint][m] - minValue) / (maxValue - minValue);

                    points.push_back(defaulttype::Vector3(xto[indexPoint][0],xto[indexPoint][1],xto[indexPoint][2]));
                    colors.push_back(defaulttype::Vec<4,float>(color, 0.0, 0.0,1.0));
                }
            }
            vparams->drawTool()->drawTriangles(points, normals, colors);
        }
        else // Show by points
        {
            glPointSize( 10);
            glBegin( GL_POINTS);
            for ( unsigned int i = 0; i < xto.size(); i++)
            {
                double color = 0;

                for ( unsigned int m=0 ; m<nbref && m_weights[i][m]>0.; m++ )
                    if(index[i][m]==showFromIndex.getValue())
                        color = (m_weights[i][m] - minValue) / (maxValue - minValue);

                glColor3f( color, 0.0, 0.0);
                glVertex3f( xto[i][0], xto[i][1], xto[i][2]);
            }
            glEnd();
        }
    }

    glPopAttrib();
}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
