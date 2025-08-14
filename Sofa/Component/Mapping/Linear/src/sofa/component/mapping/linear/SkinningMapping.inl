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
#include <sofa/component/mapping/linear/SkinningMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/helper/io/Mesh.h>
#include <limits>
#include <sofa/type/Vec.h>

#include <string>
#include <iostream>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
SkinningMapping<TIn, TOut>::SkinningMapping ()
    : Inherit ()
    , d_initPos (initData (&d_initPos, "initPos", "initial child coordinates in the world reference frame." ) )
    , d_nbRef (initData (&d_nbRef, "nbRef", "Number of primitives influencing each point." ) )
    , d_index (initData (&d_index, "indices", "parent indices for each child." ) )
    , d_weight (initData (&d_weight, "weight", "influence weights of the Dofs." ) )
    , d_showFromIndex (initData (&d_showFromIndex, ( unsigned int ) 0, "showFromIndex", "Displayed From Index." ) )
    , d_showWeights (initData (&d_showWeights, false, "showWeights", "Show influence." ) )
{
    type::vector<unsigned int> defaultNbRef;
    defaultNbRef.push_back((unsigned ) 4);

    d_nbRef.setValue(defaultNbRef);
}


template <class TIn, class TOut>
SkinningMapping<TIn, TOut>::~SkinningMapping ()
{
}



template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::init()
{
    unsigned int numChildren = this->toModel->getSize();
    sofa::helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::vec_id::read_access::position));
    sofa::helper::WriteAccessor<Data<OutVecCoord> > initPos(this->d_initPos);

    if(this->d_initPos.getValue().size() != numChildren )
    {
        initPos.resize(out.size());
        for(unsigned int i=0; i<out.size(); i++ )
            initPos[i] = out[i];
    }

    if(this->d_weight.getValue().size() != numChildren || this->d_index.getValue().size() != numChildren)
        updateWeights(); // if not defined by user -> recompute based on euclidean distances

    reinit();
    Inherit::init();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::reinit()
{
    unsigned int nbref = d_nbRef.getValue()[0];

    sofa::helper::ReadAccessor<Data<OutVecCoord> > out (*this->toModel->read(core::vec_id::read_access::position));
    sofa::helper::ReadAccessor<Data<OutVecCoord> > xto (this->d_initPos);
    sofa::helper::ReadAccessor<Data<InVecCoord> > xfrom = *this->fromModel->read(core::vec_id::read_access::restPosition);
    sofa::helper::WriteAccessor<Data<type::vector<sofa::type::SVector<InReal> > > > m_weights  (d_weight );
    const sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<unsigned int> > > > index ( this->d_index );

    msg_info() << "reinit : use d_nbRef with size = " << d_nbRef.getValue().size() << " - initpos size = " << xto.size();

    // normalize weights
    for (unsigned int i=0; i<xto.size(); i++ )
    {
        if(d_nbRef.getValue().size() == m_weights.size())
            nbref = d_nbRef.getValue()[i];

        InReal w=0;  for (unsigned int j=0; j<nbref; j++ ) w+=m_weights[i][j];
        if(w!=0) for (unsigned int j=0; j<nbref; j++ ) m_weights[i][j]/=w;
    }

    // precompute local/rotated positions
    {
        _J.resizeBlocks(out.size(), xfrom.size());
        f_localPos.resize(out.size());
        f_rotatedPos.resize(out.size());

        for(unsigned int i=0; i<out.size(); i++ )
        {
            if(d_nbRef.getValue().size() == m_weights.size())
                nbref = d_nbRef.getValue()[i];

            sofa::type::Vec<3,InReal> cto; Out::get( cto[0],cto[1],cto[2], xto[i] );
            f_localPos[i].resize(nbref);
            f_rotatedPos[i].resize(nbref);

            for (unsigned int j=0 ; j<nbref; j++ )
            {
                f_localPos[i][j]= xfrom[index[i][j]].unprojectPoint(cto) * m_weights[i][j];
                f_rotatedPos[i][j]= (cto - xfrom[index[i][j]].getCenter() ) * m_weights[i][j];
            }
        }
    }

}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::updateWeights ()
{
    msg_info() << "UPDATE WEIGHTS";

    sofa::helper::ReadAccessor<Data<OutVecCoord> > xto (this->d_initPos);
    sofa::helper::ReadAccessor<Data<InVecCoord> > xfrom = *this->fromModel->read(core::vec_id::read_access::restPosition);
    sofa::helper::WriteAccessor<Data<type::vector<sofa::type::SVector<InReal> > > > m_weights  (d_weight );
    sofa::helper::WriteAccessor<Data<type::vector<sofa::type::SVector<unsigned int> > > > index (d_index );

    index.resize( xto.size() );
    m_weights.resize ( xto.size() );
    unsigned int nbref=d_nbRef.getValue()[0];

    // compute 1/d^2 weights with Euclidean distance
    for (unsigned int i=0; i<xto.size(); i++ )
    {
        sofa::type::Vec<3,InReal> cto; Out::get( cto[0],cto[1],cto[2], xto[i] );

        // get the d_nbRef closest primitives
        index[i].resize( nbref );
        m_weights[i].resize ( nbref );
        for (unsigned int j=0; j<nbref; j++ )
        {
            m_weights[i][j]=0;
            index[i][j]=0;
        }
        for (unsigned int j=0; j<xfrom.size(); j++ )
        {
            sofa::type::Vec<3,InReal> cfrom; In::get( cfrom[0],cfrom[1],cfrom[2], xfrom[j] );
            InReal w=(cto-cfrom)*(cto-cfrom);
            if(w!=0) w=1.0f/w;
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
void SkinningMapping<TIn, TOut>::setWeights(const type::vector<sofa::type::SVector<InReal> >& weights, const type::vector<sofa::type::SVector<unsigned int> >& indices, const type::vector<unsigned int>& nbrefs)
{
    d_index = indices;
    d_weight = weights;
    d_nbRef = nbrefs;
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::apply( const sofa::core::MechanicalParams* mparams, OutDataVecCoord& outData, const InDataVecCoord& inData)
{
    SOFA_UNUSED(mparams);

    OutVecCoord& out = *outData.beginEdit();
    const InVecCoord& in = inData.getValue();

    unsigned int nbref=d_nbRef.getValue()[0];
    sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<InReal> > > > m_weights  ( this->d_weight );
    const sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<unsigned int> > > > index (d_index );
    MatBlock matblock;
    {
        _J.clear();
        for ( unsigned int i = 0 ; i < out.size(); i++ )
        {
            out[i] = OutCoord ();

            if(d_nbRef.getValue().size() == m_weights.size())
                nbref = d_nbRef.getValue()[i];

            _J.beginBlockRow(i);
            for ( unsigned int j=0; j<nbref && m_weights[i][j]>0.; j++ )
            {
                f_rotatedPos[i][j]=in[index[i][j]].rotate(f_localPos[i][j]);
                out[i] += in[index[i][j]].getCenter() * m_weights[i][j] + f_rotatedPos[i][j];

                // update the Jacobian Matrix
                //                Real w=m_weights[i][j];
                matblock(0,0) = (Real) m_weights[i][j]         ;    matblock(1,0) = (Real) 0                      ;    matblock(2,0) = (Real) 0                      ;
                matblock(0,1) = (Real) 0                       ;    matblock(1,1) = (Real) m_weights[i][j]        ;    matblock(2,1) = (Real) 0                      ;
                matblock(0,2) = (Real) 0                       ;    matblock(1,2) = (Real) 0                      ;    matblock(2,2) = (Real) m_weights[i][j]        ;
                matblock(0,3) = (Real) 0                       ;    matblock(1,3) = (Real)-f_rotatedPos[i][j][2]  ;    matblock(2,3) = (Real) f_rotatedPos[i][j][1]  ;
                matblock(0,4) = (Real) f_rotatedPos[i][j][2]   ;    matblock(1,4) = (Real) 0                      ;    matblock(2,4) = (Real)-f_rotatedPos[i][j][0]  ;
                matblock(0,5) = (Real)-f_rotatedPos[i][j][1]   ;    matblock(1,5) = (Real) f_rotatedPos[i][j][0]  ;    matblock(2,5) = (Real) 0                      ;
                _J.createBlock(index[i][j],matblock);
            }
            _J.endBlockRow();
        }
        _J.compress();
    }
    outData.endEdit();
}


template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJ( const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& outData, const InDataVecDeriv& inData)
{
    SOFA_UNUSED(mparams);

    OutVecDeriv& out = *outData.beginWriteOnly();
    const InVecDeriv& in = inData.getValue();

    unsigned int nbref=d_nbRef.getValue()[0];
    sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<InReal> > > > m_weights  (d_weight );
    const sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<unsigned int> > > > index (d_index );
    {
        for( size_t i=0 ; i<out.size() ; ++i)
        {
            out[i] = OutDeriv();

            if(d_nbRef.getValue().size() == m_weights.size())
                nbref = d_nbRef.getValue()[i];

            for ( unsigned int j=0; j<nbref && m_weights[i][j]>0.; j++ )
            {
                out[i] += getLinear( in[index[i][j]] ) * m_weights[i][j] + cross(getAngular(in[index[i][j]]), f_rotatedPos[i][j]);
            }
        }
    }

    outData.endEdit();
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJT( const sofa::core::MechanicalParams* mparams, InDataVecDeriv& outData, const OutDataVecDeriv& inData)
{
    SOFA_UNUSED(mparams);

    InVecDeriv& out = *outData.beginEdit();
    const OutVecDeriv& in = inData.getValue();

    unsigned int nbref=d_nbRef.getValue()[0];
    sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<InReal> > > > m_weights  (d_weight );
    const sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<unsigned int> > > > index (d_index );

    for( size_t i=0 ; i<index.size() ; ++i)
    {
        if(d_nbRef.getValue().size() == m_weights.size())
            nbref = d_nbRef.getValue()[i];

        for ( unsigned int j=0; j<nbref && m_weights[i][j]>0.; j++ )
        {
            getLinear(out[index[i][j]])  += in[i] * m_weights[i][j];
            getAngular(out[index[i][j]]) += cross(f_rotatedPos[i][j], in[i]);
        }
    }

    outData.endEdit();
}



template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::applyJT ( const sofa::core::ConstraintParams* cparams, InDataMatrixDeriv& outData, const OutDataMatrixDeriv& inData)
{
    SOFA_UNUSED(cparams);

    InMatrixDeriv& parentJacobians = *outData.beginEdit();
    const OutMatrixDeriv& childJacobians = inData.getValue();

    unsigned int nbref=d_nbRef.getValue()[0];
    sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<InReal> > > > m_weights  (d_weight );
    const sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<unsigned int> > > > index (d_index );

    for (typename Out::MatrixDeriv::RowConstIterator childJacobian = childJacobians.begin(); childJacobian != childJacobians.end(); ++childJacobian)
    {
        typename In::MatrixDeriv::RowIterator parentJacobian = parentJacobians.writeLine(childJacobian.index());

        for (typename Out::MatrixDeriv::ColConstIterator childParticle = childJacobian.begin(); childParticle != childJacobian.end(); ++childParticle)
        {
            unsigned int childIndex = childParticle.index();
            const OutDeriv& childJacobianVec = childParticle.val();

            if(d_nbRef.getValue().size() == m_weights.size())
                nbref = d_nbRef.getValue()[childIndex];

            for ( unsigned int j=0; j<nbref && m_weights[childIndex][j]>0.; j++ )
            {
                InDeriv parentJacobianVec;
                getLinear(parentJacobianVec)  += childJacobianVec * m_weights[childIndex][j];
                getAngular(parentJacobianVec) += cross(f_rotatedPos[childIndex][j], childJacobianVec);
                parentJacobian.addCol(index[childIndex][j],parentJacobianVec);
            }
        }
    }
    outData.endEdit();
}

template <class TIn, class TOut>
const sofa::type::vector<sofa::linearalgebra::BaseMatrix*>* SkinningMapping<TIn, TOut>::getJs()
{
    return new sofa::type::vector<sofa::linearalgebra::BaseMatrix*>(1, (sofa::linearalgebra::BaseMatrix*)&_J);
}

template <class TIn, class TOut>
const  sofa::linearalgebra::BaseMatrix* SkinningMapping<TIn, TOut>::getJ()
{
    return (sofa::linearalgebra::BaseMatrix*)&_J;
}

template <class TIn, class TOut>
void SkinningMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    const typename Out::VecCoord& xto = this->toModel->read(core::vec_id::read_access::position)->getValue();
    const typename In::VecCoord& xfrom = this->fromModel->read(core::vec_id::read_access::position)->getValue();
    unsigned int nbref = this->d_nbRef.getValue()[0];

    sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<InReal> > > > m_weights  (d_weight );
    const sofa::helper::ReadAccessor<Data<type::vector<sofa::type::SVector<unsigned int> > > > index (d_index );

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    std::vector<sofa::type::RGBAColor> colorVector;
    std::vector<sofa::type::Vec3> vertices;

    if ( vparams->displayFlags().getShowMappings() )
    {
        // Display mapping links between in and out elements

        for ( unsigned int i=0; i<xto.size(); i++ )
        {
            if(d_nbRef.getValue().size() == m_weights.size())
                nbref = d_nbRef.getValue()[i];

            for ( unsigned int m=0 ; m<nbref && m_weights[i][m]>0.; m++ )
            {
                colorVector.emplace_back( m_weights[i][m],m_weights[i][m], 0.f, 1.f );
                vertices.push_back(sofa::type::Vec3( xfrom[index[i][m]].getCenter() ));
                vertices.push_back(sofa::type::Vec3( xto[i] ));
            }
        }
        vparams->drawTool()->drawLines(vertices,1,colorVector);
        vertices.clear();
        colorVector.clear();
    }

    // Show weights
    if ( d_showWeights.getValue())
    {
        InReal minValue = 1, maxValue = 0;


        if ( ! triangles.empty()) // Show on mesh
        {
            std::vector< type::Vec3 > points;
            const std::vector< type::Vec3 > normals;
            std::vector<sofa::type::RGBAColor> colors;
            for ( unsigned int i = 0; i < triangles.size(); i++)
            {
                for ( unsigned int j = 0; j < 3; j++)
                {
                    const unsigned int& indexPoint = triangles[i][j];
                    double color = 0;

                    if(d_nbRef.getValue().size() == m_weights.size())
                        nbref = d_nbRef.getValue()[indexPoint];

                    for ( unsigned int m=0 ; m<nbref && m_weights[indexPoint][m]>0.; m++ )
                        if(index[indexPoint][m] == d_showFromIndex.getValue())
                            color = (m_weights[indexPoint][m] - minValue) / (maxValue - minValue);

                    points.push_back(type::Vec3(xto[indexPoint][0],xto[indexPoint][1],xto[indexPoint][2]));
                    colors.push_back({ float(color), 0.0f, 0.0f, 1.0f });
                }
            }
            vparams->drawTool()->drawTriangles(points, normals, colors);
        }
        else // Show by points
        {
            for ( unsigned int i = 0; i < xto.size(); i++)
            {
                double color = 0;

                if(d_nbRef.getValue().size() == m_weights.size())
                    nbref = d_nbRef.getValue()[i];

                for ( unsigned int m=0 ; m<nbref && m_weights[i][m]>0.; m++ )
                    if(index[i][m] == d_showFromIndex.getValue())
                        color = (m_weights[i][m] - minValue) / (maxValue - minValue);

                colorVector.push_back(sofa::type::RGBAColor( color, 0.0, 0.0, 1.0 ));
                vertices.push_back( sofa::type::Vec3(xto[i][0], xto[i][1], xto[i][2]));
            }
            vparams->drawTool()->drawPoints(vertices,10,colorVector);
        }
    }

}



} //namespace sofa::component::mapping::linear
