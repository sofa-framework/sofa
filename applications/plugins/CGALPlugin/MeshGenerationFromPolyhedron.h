/*
 * MeshGenerationFromPolyhedron.h
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */

#ifndef CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_H
#define CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_H

#define CGAL_MESH_3_VERBOSE

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/visual/VisualParams.h>

#include <CGAL/version.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>


namespace cgal
{

template <class DataTypes>
class MeshGenerationFromPolyhedron : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MeshGenerationFromPolyhedron,DataTypes),sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::Tetra Tetra;

    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;


public:
    MeshGenerationFromPolyhedron();
    virtual ~MeshGenerationFromPolyhedron() { }

    void init();
    void reinit();

    void update();

    void draw(const sofa::core::visual::VisualParams* vparams);

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

    Data<bool> frozen;

    //Parameters
    Data<double> facetAngle, facetSize, facetApproximation;
    Data<double> cellRatio, cellSize;
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,8,0)
    Data<double> sharpEdgeAngle;
    Data<double> sharpEdgeSize;
#endif
    Data<bool> odt, lloyd, perturb, exude;
    Data<int> odt_max_it, lloyd_max_it;
    Data<double> perturb_max_time, exude_max_time;

    // Display
    Data<bool> drawTetras;
    Data<bool> drawSurface;

    // A modifier creating a triangle with the incremental builder.
    template <class HDS>
    class AddTriangles : public CGAL::Modifier_base<HDS>
    {
    public:
        const VecCoord& points;
        const SeqTriangles& triangles;
        const SeqQuads& quads;

        AddTriangles(const VecCoord& points, const SeqTriangles& triangles, const SeqQuads& quads)
            : points(points), triangles(triangles), quads(quads) {}

        void operator()( HDS& hds)
        {
            typedef typename HDS::Vertex   Vertex;
            typedef typename Vertex::Point CPoint;
            // Postcondition: `hds' is a valid polyhedral surface.
            CGAL::Polyhedron_incremental_builder_3<HDS> polyhedronBuilder( hds, true);

            if (!triangles.empty() || !quads.empty())
            {
                //we assume that the point iterator gives point in ascendant order (0,.. n+1...)
                //std::map<int, Vertex_handle> s2cVertices;

                //polyhedronBuilder.begin_surface(points.size(), triangles.size()+quads.size());
                polyhedronBuilder.begin_surface(points.size(), triangles.size()+2*quads.size());

                for (typename VecCoord::const_iterator itVertex = points.begin() ; itVertex != points.end() ; ++itVertex)
                {
                    Point p = (*itVertex);
                    polyhedronBuilder.add_vertex( CPoint(p[0], p[1], p[2]));
                }

                for (SeqTriangles::const_iterator itTriangle = triangles.begin() ; itTriangle != triangles.end() ; ++itTriangle)
                {
                    Triangle t = (*itTriangle);

                    polyhedronBuilder.begin_facet();
                    polyhedronBuilder.add_vertex_to_facet( t[0]);
                    polyhedronBuilder.add_vertex_to_facet( t[1]);
                    polyhedronBuilder.add_vertex_to_facet( t[2]);
                    polyhedronBuilder.end_facet();

                }
                for (SeqQuads::const_iterator itQuad = quads.begin() ; itQuad != quads.end() ; ++itQuad)
                {
                    Quad t = (*itQuad);

                    polyhedronBuilder.begin_facet();
                    polyhedronBuilder.add_vertex_to_facet( t[0]);
                    polyhedronBuilder.add_vertex_to_facet( t[1]);
                    polyhedronBuilder.add_vertex_to_facet( t[2]);
                    // polyhedronBuilder.add_vertex_to_facet( t[3]);
                    polyhedronBuilder.end_facet();

                    polyhedronBuilder.begin_facet();
                    polyhedronBuilder.add_vertex_to_facet( t[0]);
                    polyhedronBuilder.add_vertex_to_facet( t[2]);
                    polyhedronBuilder.add_vertex_to_facet( t[3]);
                    polyhedronBuilder.end_facet();

                }
                if ( polyhedronBuilder.check_unconnected_vertices() )
                {
                    std::cout << "Remove unconnected vertices" << std::endl;
                    polyhedronBuilder.remove_unconnected_vertices();
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
