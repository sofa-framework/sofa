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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTOPOLOGYCONTAINER_INL
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTOPOLOGYCONTAINER_INL
#include <sofa/core/visual/VisualParams.h>

#include "BarycentricMapperTopologyContainer.h"

namespace sofa
{

namespace component
{

namespace mapping
{

namespace _barycentricmappertopologycontainer_
{

using defaulttype::Vec3d;

template <class In, class Out, class MappingDataType, class Element>
BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::BarycentricMapperTopologyContainer(core::topology::BaseMeshTopology* fromTopology,
                                                                                                       topology::PointSetTopologyContainer* toTopology)
     : Inherit1(fromTopology, toTopology),
       d_map(initData(&d_map,"map", "mapper data")),
       m_matrixJ(NULL),
       m_updateJ(true)
 {}



template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::clear(int size)
{
    helper::vector<MappingDataType>& vectorData = *(d_map.beginEdit());
    vectorData.clear();
    if ( size>0 ) vectorData.reserve ( size );
    d_map.endEdit();
}


template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::init ( const typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const helper::vector<Element>& elements = getElements();
    helper::vector<Mat3x3d> bases;
    helper::vector<Vector3> centers;

    this->clear ( (int)out.size() );
    bases.resize ( elements.size() );
    centers.resize ( elements.size() );

    // Compute bases and centers of each element
    for ( unsigned int e = 0; e < elements.size(); e++ )
    {
        Element element = elements[e];

        Mat3x3d base;
        computeBase(base,in,element);
        bases[e] = base;

        Vector3 center;
        computeCenter(center,in,element);
        centers[e] = center;
    }

    // Compute distances to get nearest element and corresponding bary coef
    for ( unsigned int i=0; i<out.size(); i++ )
    {
        Vec3d outPos = Out::getCPos(out[i]);
        Vector3 baryCoords;
        int elementIndex = -1;
        double distance = 1e10;
        for ( unsigned int e = 0; e < elements.size(); e++ )
        {
            Vec3d bary = bases[e] * ( outPos - in[elements[e][0]] );
            double dist;
            computeDistance(dist, bary);
            if ( dist>0 )
                dist = ( outPos-centers[e] ).norm2();
            if ( dist<distance )
            {
                baryCoords = bary;
                distance = dist;
                elementIndex = e;
            }
        }
        addPointInElement(elementIndex, baryCoords.ptr());
    }
}


template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::applyJT ( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in )
{
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();
    const helper::vector< Element >& elements = getElements();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

                const Element& element = elements[d_map.getValue()[indexIn].in_index];

                helper::vector<SReal> baryCoef = getBaryCoef(d_map.getValue()[indexIn].baryCoords);
                for (unsigned int j=0; j<element.size(); j++)
                    o.addCol(element[j], data*baryCoef[j]);
            }
        }
    }
}



template <class In, class Out, class MappingDataType, class Element>
const sofa::defaulttype::BaseMatrix* BarycentricMapperTopologyContainer<In,Out,MappingDataType, Element>::getJ(int outSize, int inSize)
{
    if (m_matrixJ && !m_updateJ)
        return m_matrixJ;

    if (!m_matrixJ) m_matrixJ = new MatrixType;
    if (m_matrixJ->rowBSize() != (MatrixTypeIndex)outSize || m_matrixJ->colBSize() != (MatrixTypeIndex)inSize)
        m_matrixJ->resize(outSize*NOut, inSize*NIn);
    else
        m_matrixJ->clear();

    return m_matrixJ;

    const helper::vector<Element>& elements = getElements();

    for( size_t outId=0 ; outId<this->maskTo->size() ; ++outId)
    {
        if( !this->maskTo->getEntry(outId) ) continue;

        const Element& element = elements[d_map.getValue()[outId].in_index];

        helper::vector<SReal> baryCoef = getBaryCoef(d_map.getValue()[outId].baryCoords);
        for (unsigned int j=0; j<element.size(); j++)
            this->addMatrixContrib(m_matrixJ, outId, element[j], baryCoef[j]);
    }

    m_matrixJ->compress();
    m_updateJ = false;
    return m_matrixJ;
}


template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::applyJT ( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    const helper::vector<Element>& elements = getElements();

    ForceMask& mask = *this->maskFrom;
    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->getEntry(i) ) continue;

        int index = d_map.getValue()[i].in_index;
        const Element& element = elements[index];

        const typename Out::DPos inPos = Out::getDPos(in[i]);
        helper::vector<SReal> baryCoef = getBaryCoef(d_map.getValue()[i].baryCoords);
        for (unsigned int j=0; j<element.size(); j++)
        {
            out[element[j]] += inPos * baryCoef[j];
            mask.insertEntry(element[j]);
        }
    }
}

template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::applyJ ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    out.resize( d_map.getValue().size() );

    const helper::vector<Element>& elements = getElements();

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->isActivated() && !this->maskTo->getEntry(i) ) continue;

        int index = d_map.getValue()[i].in_index;
        const Element& element = elements[index];

        helper::vector<SReal> baryCoef = getBaryCoef(d_map.getValue()[i].baryCoords);
        InDeriv inPos{0.,0.,0.};
        for (unsigned int j=0; j<element.size(); j++)
            inPos += in[element[j]] * baryCoef[j];

        Out::setDPos(out[i] , inPos);
    }
}


template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::resize( core::State<Out>* toModel )
{
    toModel->resize(d_map.getValue().size());
}

template<class In, class Out, class MappingDataType, class Element>
bool BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::isEmpty()
{
    return d_map.getValue().empty();
}



template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::apply ( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    out.resize( d_map.getValue().size() );

    const helper::vector<Element>& elements = getElements();
    for ( unsigned int i=0; i<d_map.getValue().size(); i++ )
    {
        int index = d_map.getValue()[i].in_index;
        const Element& element = elements[index];

        helper::vector<SReal> baryCoef = getBaryCoef(d_map.getValue()[i].baryCoords);
        InDeriv inPos{0.,0.,0.};
        for (unsigned int j=0; j<element.size(); j++)
            inPos += in[element[j]] * baryCoef[j];

        Out::setCPos(out[i] , inPos);
    }
}


template <class In, class Out, class MappingDataType, class Element>
void BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>::draw  (const core::visual::VisualParams* vparams,
                                                                                const typename Out::VecCoord& out,
                                                                                const typename In::VecCoord& in )
{
    // Draw line between mapped node (out) and nodes of nearest element (in)
    const helper::vector<Element>& elements = getElements();

    std::vector< Vector3 > points;
    {
        for ( unsigned int i=0; i<d_map.getValue().size(); i++ )
        {
            int index = d_map.getValue()[i].in_index;
            const Element& element = elements[index];
            helper::vector<SReal> baryCoef = getBaryCoef(d_map.getValue()[i].baryCoords);
            for ( unsigned int j=0; j<element.size(); j++ )
            {
                if ( baryCoef[j]<=-0.0001 || baryCoef[j]>=0.0001 )
                {
                    points.push_back ( Out::getCPos(out[i]) );
                    points.push_back ( in[element[j]] );
                }
            }
        }
    }
    vparams->drawTool()->drawLines ( points, 1, sofa::defaulttype::Vec<4,float> ( 0,1,0,1 ) );
}

template<class In, class Out, class MappingData, class Element>
std::istream& operator >> ( std::istream& in, BarycentricMapperTopologyContainer<In, Out, MappingData, Element> &b )
{
    unsigned int size_vec;

    in >> size_vec;
    sofa::helper::vector<MappingData>& m = *(b.d_map.beginEdit());
    m.clear();

    MappingData value;
    for (unsigned int i=0; i<size_vec; i++)
    {
        in >> value;
        m.push_back(value);
    }
    b.d_map.endEdit();
    return in;
}

template<class In, class Out, class MappingData, class Element>
std::ostream& operator << ( std::ostream& out, const BarycentricMapperTopologyContainer<In, Out, MappingData, Element> & b )
{
    out << b.d_map.getValue().size();
    out << " " ;
    out << b.d_map;

    return out;
}



}
} //
} //
} //

#endif
