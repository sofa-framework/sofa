/*
 * CuboidMesh.h
 *
 *  Created on: 12 sep. 2011
 *      Author: Yiyi
 */

#ifndef CGALPLUGIN_CUBOIDMESH_H
#define CGALPLUGIN_CUBOIDMESH_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/gl/template.h>

#include <math.h>
#include   <algorithm>

namespace cgal
{

template <class DataTypes>
class CuboidMesh : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CuboidMesh,DataTypes),sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef sofa::helper::fixed_array<int, 3> Index;
    //    typedef sofa::helper::vector<Real> VecReal;

    //        typedef sofa::core::topology::BaseMeshTopology::PointID PointID;
    //    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    //    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    //    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::Tetra Tetra;
    typedef sofa::core::topology::BaseMeshTopology::Hexa Hexa;


    //    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    //    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    //    typedef sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;


public:
    CuboidMesh();
    virtual ~CuboidMesh() { };

    void init();
    void reinit();

    void update();
    void orientate();
    void draw();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const CuboidMesh<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Inputs
    Data<unsigned> m_debug;
    Data<double> m_length;
    Data<double> m_height;
    Data<int> m_number;
    Data<bool> m_viewPoints;
    Data<bool> m_viewTetras;

    //Outputs
    Data<VecCoord> m_points;
    Data<SeqTetrahedra> m_tetras;

    //Parameters
    unsigned m_nbVertices, m_nbCenters;
    unsigned m_nbBdCenters_i, m_nbBdCenters_j, m_nbBdCenters_k;
    unsigned m_nbTetras_i, m_nbTetras_j, m_nbTetras_k;
    int n, m, a;
    Real l, h, dl, t;
    std::map<Index, unsigned> m_ptID;
    unsigned short debug;


};

#if defined(WIN32) && !defined(CGALPLUGIN_CUBOIDMESH_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API CuboidMesh<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API CuboidMesh<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} //cgal

#endif /* CGALPLUGIN_CUBOIDMESH_H */
