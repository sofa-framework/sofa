#ifndef SOFA_COMPONENTS_BARYCENTRICMAPPING_H
#define SOFA_COMPONENTS_BARYCENTRICMAPPING_H

#include "Sofa/Core/MechanicalMapping.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "MeshTopology.h"
#include "RegularGridTopology.h"
#include <vector>

namespace Sofa
{

namespace Components
{

template <class BaseMapping>
class BarycentricMapping : public BaseMapping, public Abstract::VisualModel
{
public:
    typedef BaseMapping Inherit;
    typedef typename BaseMapping::In In;
    typedef typename BaseMapping::Out Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename InCoord::value_type Real;
    typedef typename OutCoord::value_type OutReal;

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

    class Mapper
    {
    public:
        virtual ~Mapper() { }
        virtual void apply( typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in ) = 0;
        virtual void applyJ( typename BaseMapping::Out::VecDeriv& out, const typename BaseMapping::In::VecDeriv& in ) = 0;
        virtual void applyJT( typename BaseMapping::In::VecDeriv& out, const typename BaseMapping::Out::VecDeriv& in ) = 0;
        virtual void draw( const typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in) = 0;
    };

    class RegularGridMapper : public Mapper
    {
    public:
        RegularGridMapper(RegularGridTopology* topology) : topology(topology) {}

        std::vector<CubeData> map;
        RegularGridTopology* topology;
        void apply( typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in );
        void applyJ( typename BaseMapping::Out::VecDeriv& out, const typename BaseMapping::In::VecDeriv& in );
        void applyJT( typename BaseMapping::In::VecDeriv& out, const typename BaseMapping::Out::VecDeriv& in );
        void draw( const typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in);
    };

    class MeshMapper : public Mapper
    {
    public:
        MeshMapper(MeshTopology* topology) : topology(topology) {}

        std::vector< MappingData<1,0> > map1d;
        std::vector< MappingData<2,0> > map2d;
        std::vector< MappingData<3,0> > map3d;
        MeshTopology* topology;

        void clear();

        void addPointInLine(const OutCoord& p, int lineIndex, const Real* baryCoords);

        void addPointInTriangle(const OutCoord& p, int triangleIndex, const Real* baryCoords);
        void createPointInTriangle(const OutCoord& p, int triangleIndex, const InVecCoord* points);

        void addPointInQuad(const OutCoord& p, int quadIndex, const Real* baryCoords);
        void createPointInQuad(const OutCoord& p, int quadIndex, const InVecCoord* points);

        void addPointInTetra(const OutCoord& p, int tetraIndex, const Real* baryCoords);

        void addPointInCube(const OutCoord& p, int cubeIndex, const Real* baryCoords);

        void apply( typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in );
        void applyJ( typename BaseMapping::Out::VecDeriv& out, const typename BaseMapping::In::VecDeriv& in );
        void applyJT( typename BaseMapping::In::VecDeriv& out, const typename BaseMapping::Out::VecDeriv& in );
        void draw( const typename BaseMapping::Out::VecCoord& out, const typename BaseMapping::In::VecCoord& in);
    };

protected:
    //std::vector<MapCoef> map;

    //std::vector<LineData> mapLines;
    //std::vector<TriangleData> mapTriangles;
    //std::vector<QuadData> mapQuads;
    //std::vector<TetraData> mapTetras;
    //std::vector<CubeData> mapCubes;

    Mapper* mapper;

    void calcMap(RegularGridTopology* topo);

    void calcMap(MeshTopology* topo);

public:
    BarycentricMapping(In* from, Out* to)
        : Inherit(from, to), mapper(NULL)
    {
    }

    BarycentricMapping(In* from, Out* to, Mapper* mapper)
        : Inherit(from, to), mapper(mapper)
    {
    }

    virtual ~BarycentricMapping()
    {
        if (mapper!=NULL)
            delete mapper;
    }

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
