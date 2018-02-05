/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/Simulation.h>
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
    sofa::core::objectmodel::Data<VecCoord> f_X0;
    sofa::core::objectmodel::Data<SeqTriangles> f_triangles;
    sofa::core::objectmodel::Data<SeqQuads> f_quads;

    //Outputs
    sofa::core::objectmodel::Data<VecCoord> f_newX0;
    sofa::core::objectmodel::Data<SeqTetrahedra> f_tetrahedra;

    sofa::core::objectmodel::Data<bool> frozen;

    //Parameters
    sofa::core::objectmodel::Data<double> facetAngle, facetSize, facetApproximation;
    sofa::core::objectmodel::Data<double> cellRatio, cellSize;
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,8,0)
    sofa::core::objectmodel::Data<double> sharpEdgeAngle;
    sofa::core::objectmodel::Data<double> sharpEdgeSize;
#endif
    sofa::core::objectmodel::Data<bool> odt, lloyd, perturb, exude;
    sofa::core::objectmodel::Data<int> odt_max_it, lloyd_max_it;
    sofa::core::objectmodel::Data<double> perturb_max_time, exude_max_time;
    sofa::core::objectmodel::Data<int> ordering;
    sofa::core::objectmodel::Data<bool> constantMeshProcess;
    sofa::core::objectmodel::Data<unsigned int> meshingSeed;

    // Display
    sofa::core::objectmodel::Data<bool> drawTetras;
    sofa::core::objectmodel::Data<bool> drawSurface;

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

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CGALPLUGIN_API MeshGenerationFromPolyhedron<sofa::defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_CGALPLUGIN_API MeshGenerationFromPolyhedron<sofa::defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} //cgal

#endif /* CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_H */
