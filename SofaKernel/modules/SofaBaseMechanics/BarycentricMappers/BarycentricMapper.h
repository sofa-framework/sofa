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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPER_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPER_H

#include <sofa/core/Mapping.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace mapping
{

namespace _barycentricmapper_
{

using core::visual::VisualParams;
using sofa::defaulttype::BaseMatrix;
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3Types;

/// Base class for barycentric mapping topology-specific mappers
template<class In, class Out>
class BarycentricMapper : public virtual core::objectmodel::BaseObject
{

public:

    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapper,In,Out),core::objectmodel::BaseObject);

    typedef typename In::Real Real;
    typedef typename In::Real InReal;
    typedef typename Out::Real OutReal;

    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Deriv InDeriv;

    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Deriv OutDeriv;

    enum { NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size };
    enum { NOut = sofa::defaulttype::DataTypeInfo<OutDeriv>::Size };
    typedef defaulttype::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<MBloc> MatrixType;


public:

    using BaseObject::init;
    virtual void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) = 0;

    using BaseObject::draw;
    virtual void draw(const VisualParams*, const typename Out::VecCoord& out, const typename In::VecCoord& in) = 0;

    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) = 0;
    virtual const BaseMatrix* getJ(int outSize, int inSize);
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) = 0;
    virtual void applyOnePoint( const unsigned int& hexaId, typename Out::VecCoord& out, const typename In::VecCoord& in);
    virtual void clear( int reserve=0 ) =0;

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapper< In, Out > & ) {return in;}
    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapper< In, Out > &  ) { return out; }


protected:
    void addMatrixContrib(MatrixType* m, int row, int col, Real value);

    template< int NC,  int NP>
    class MappingData
    {
    public:
        int in_index;
        Real baryCoords[NC];

        inline friend std::istream& operator >> ( std::istream& in, MappingData< NC, NP> &m )
        {
            in>>m.in_index;
            for (int i=0; i<NC; i++) in >> m.baryCoords[i];
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const MappingData< NC , NP > & m )
        {
            out << m.in_index;
            for (int i=0; i<NC; i++)
                out << " " << m.baryCoords[i];
            out << "\n";
            return out;
        }

    };


public:
    typedef MappingData<1,2> LineData;
    typedef MappingData<2,3> TriangleData;
    typedef MappingData<2,4> QuadData;
    typedef MappingData<3,4> TetraData;
    typedef MappingData<3,8> CubeData;
    typedef MappingData<1,0> MappingData1D;
    typedef MappingData<2,0> MappingData2D;
    typedef MappingData<3,0> MappingData3D;


protected:
    BarycentricMapper() {}
    virtual ~BarycentricMapper() override {}

private:
    BarycentricMapper(const BarycentricMapper& n) ;
    BarycentricMapper& operator=(const BarycentricMapper& n) ;
};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPER_CPP)
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3dTypes, ExtVec3Types >;


#endif

}

using _barycentricmapper_::BarycentricMapper;

}}}

#endif
