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
#include <sofa/component/mapping/linear/config.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mapping::linear::_barycentricmapper_
{

using sofa::linearalgebra::BaseMatrix;
using sofa::defaulttype::Vec3Types;

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

    static constexpr sofa::Size NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size;
    static constexpr sofa::Size NOut = sofa::defaulttype::DataTypeInfo<OutDeriv>::Size;
    
    typedef type::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<MBloc> MatrixType;

    using Index = sofa::Index;

public:

    using BaseObject::init;
    virtual void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) = 0;

    using BaseObject::draw;
    virtual void draw(const core::visual::VisualParams*, const typename Out::VecCoord& out, const typename In::VecCoord& in) = 0;

    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) = 0;
    virtual const BaseMatrix* getJ(int outSize, int inSize);
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) = 0;
    virtual void applyOnePoint( const Index& hexaId, typename Out::VecCoord& out, const typename In::VecCoord& in);
    virtual void clear( std::size_t reserve=0 ) =0;

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapper< In, Out > & ) {return in;}
    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapper< In, Out > &  ) { return out; }


protected:
    void addMatrixContrib(MatrixType* m, sofa::Index row, sofa::Index col, Real value);

    template< sofa::Size NC, sofa::Size NP>
    class MappingData
    {
    public:
        static constexpr std::size_t NumberOfCoordinates = NC;
        
        Index in_index;
        std::array<Real, NC> baryCoords;

        inline friend std::istream& operator >> ( std::istream& in, MappingData< NC, NP> &m )
        {
            in>>m.in_index;
            for (sofa::Index i=0; i<NC; i++) in >> m.baryCoords[i];
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const MappingData< NC , NP > & m )
        {
            out << m.in_index;
            for (sofa::Index i=0; i<NC; i++)
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
    ~BarycentricMapper() override {}

private:
    BarycentricMapper(const BarycentricMapper& n) ;
    BarycentricMapper& operator=(const BarycentricMapper& n) ;
};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPER_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapper< Vec3Types, Vec3Types >;
#endif

} // namespace sofa::component::mapping::linear::_barycentricmapper_


namespace sofa::component::mapping::linear
{

using _barycentricmapper_::BarycentricMapper;

} // namespace sofa::component::mapping::linear
