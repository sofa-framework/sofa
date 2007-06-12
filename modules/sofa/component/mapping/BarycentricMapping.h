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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_H

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace mapping
{

/// Base class for barycentric mapping topology-specific mappers
template<class In, class Out>
class BarycentricMapper
{
public:
    typedef typename In::Real Real;
    typedef typename Out::Real OutReal;

    template<int NC, int NP>
    class MappingData
    {
    public:
        int in_index;
        //unsigned int points[NP];
        Real baryCoords[NC];
    };

    typedef MappingData<1,2> LineData;
    typedef MappingData<2,3> TriangleData;
    typedef MappingData<2,4> QuadData;
    typedef MappingData<3,4> TetraData;
    typedef MappingData<3,8> CubeData;

    virtual ~BarycentricMapper()
    {}
    virtual void init() = 0;
    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) = 0;
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::VecConst& out, const typename Out::VecConst& in ) = 0;
    virtual void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in) = 0;
};

/// Class allowing barycentric mapping computation on a RegularGridTopology
template<class In, class Out>
class RegularGridMapper : public BarycentricMapper<In,Out>
{
public:
    typedef BarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
protected:
    std::vector<CubeData> map;
    topology::RegularGridTopology* topology;
public:
    RegularGridMapper(topology::RegularGridTopology* topology) : topology(topology)
    {}

    void clear(int reserve=0);

    int addPointInCube(int cubeIndex, const Real* baryCoords);

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);
};

/// Class allowing barycentric mapping computation on a MeshTopology
template<class In, class Out>
class MeshMapper : public BarycentricMapper<In,Out>
{
public:
    typedef BarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
protected:
    std::vector< Inherit::MappingData<1,0> > map1d;
    std::vector< Inherit::MappingData<2,0> > map2d;
    std::vector< Inherit::MappingData<3,0> > map3d;
    topology::MeshTopology* topology;

public:
    MeshMapper(topology::MeshTopology* topology) : topology(topology)
    {}

    void clear(int reserve3d=0, int reserve2d=0, int reserve1d=0);

    int addPointInLine(int lineIndex, const Real* baryCoords);
    int createPointInLine(const typename Out::Coord& p, int lineIndex, const typename In::VecCoord* points);

    int addPointInTriangle(int triangleIndex, const Real* baryCoords);
    int createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points);

    int addPointInQuad(int quadIndex, const Real* baryCoords);
    int createPointInQuad(const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points);

    int addPointInTetra(int tetraIndex, const Real* baryCoords);

    int addPointInCube(int cubeIndex, const Real* baryCoords);

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);
};

template <class BasicMapping>
class BarycentricMapping : public BasicMapping, public core::VisualModel
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename In::DataTypes InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::SparseVecDeriv InSparseVecDeriv;
    typedef typename InDataTypes::SparseDeriv InSparseDeriv;
    typedef typename InDataTypes::Real Real;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    typedef typename OutDataTypes::Coord OutCoord;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::SparseVecDeriv OutSparseVecDeriv;
    typedef typename OutDataTypes::SparseDeriv OutSparseDeriv;
    typedef typename OutDataTypes::Real OutReal;

protected:

    typedef BarycentricMapper<InDataTypes,OutDataTypes> Mapper;

    Mapper* mapper;

    void calcMap(topology::RegularGridTopology* topo);

    void calcMap(topology::MeshTopology* topo);

public:
    BarycentricMapping(In* from, Out* to)
        : Inherit(from, to), mapper(NULL)
    {}

    BarycentricMapping(In* from, Out* to, Mapper* mapper)
        : Inherit(from, to), mapper(mapper)
    {}

    virtual ~BarycentricMapping()
    {
        if (mapper!=NULL)
            delete mapper;
    }

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );

    // -- VisualModel interface
    void draw();
    void initTextures()
    { }
    void update()
    { }

protected:

    bool getShow(const core::objectmodel::BaseObject* m) const
    {
        return m->getContext()->getShowMappings();
    }

    bool getShow(const core::componentmodel::behavior::BaseMechanicalMapping* m) const
    {
        return m->getContext()->getShowMechanicalMappings();
    }

};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
