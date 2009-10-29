/*
 * MeshGenerationFromPolyhedron.h
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */

#ifndef CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_H
#define CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>


namespace cgal
{

template <class DataTypes>
class MeshGenerationFromPolyhedron : public virtual sofa::core::objectmodel::DataEngine, public virtual sofa::core::objectmodel::BaseObject
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef sofa::core::componentmodel::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::Tetra Tetra;

    typedef sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef sofa::core::componentmodel::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;


public:
    MeshGenerationFromPolyhedron();
    virtual ~MeshGenerationFromPolyhedron() { };

    void init();
    void reinit();

    void update();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const MeshGenerationFromPolyhedron<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Inputs
    Data<VecCoord> f_X0;
    Data<SeqTriangles> f_triangles;
    Data<SeqQuads> f_quads;

    //Outputs
    Data<VecCoord> f_newX0;
    Data<SeqTetrahedra> f_tetrahedra;

    //Parameters
    Data<double> facetAngle, facetSize, facetApproximation;
    Data<double> cellRatio, cellSize;

    // A modifier creating a triangle with the incremental builder.
    template <class HDS>
    class AddTriangles : public CGAL::Modifier_base<HDS>
    {
    public:
        VecCoord points;
        SeqTriangles triangles;

        AddTriangles(const VecCoord& points, const SeqTriangles& triangles)
            : points(points), triangles(triangles) {}

        void operator()( HDS& hds)
        {
            typedef typename HDS::Vertex   Vertex;
            typedef typename Vertex::Point CPoint;
            // Postcondition: `hds' is a valid polyhedral surface.
            CGAL::Polyhedron_incremental_builder_3<HDS> polyhedronBuilder( hds, true);

            if (!triangles.empty())
            {
                //we assume that the point iterator gives point in ascendant order (0,.. n+1...)
                //std::map<int, Vertex_handle> s2cVertices;

                polyhedronBuilder.begin_surface(points.size(), triangles.size());

                for (typename VecCoord::const_iterator itVertex = points.begin() ; itVertex != points.end() ; ++itVertex)
                {
                    Point p = (*itVertex);
                    polyhedronBuilder.add_vertex( CPoint(p[0], p[1], p[2]));
                }

                for (SeqTriangles::const_iterator itTriangle = triangles.begin() ; itTriangle != triangles.end() ; ++itTriangle)
                {
                    Triangle t = (*itTriangle);

                    polyhedronBuilder.begin_facet();
                    polyhedronBuilder.add_vertex_to_facet( t[2]);
                    polyhedronBuilder.add_vertex_to_facet( t[0]);
                    polyhedronBuilder.add_vertex_to_facet( t[1]);

                    polyhedronBuilder.end_facet();

                }
                if ( polyhedronBuilder.check_unconnected_vertices() )
                {
                    polyhedronBuilder.remove_unconnected_vertices();
                    std::cout << "Remove unconnect vertices" << std::endl;
                }


                polyhedronBuilder.end_surface();
            }
        }
    };

};

#if defined(WIN32) && !defined(CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API MeshGenerationFromPolyhedron<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API MeshGenerationFromPolyhedron<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} //cgal

#endif /* CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_H */
