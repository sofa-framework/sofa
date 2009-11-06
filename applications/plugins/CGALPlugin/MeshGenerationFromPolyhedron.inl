/*
 * MeshGenerationFromPolyhedron.inl
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */
#ifndef CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_INL
#define CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_INL
#include "MeshGenerationFromPolyhedron.h"

#include <CGAL/AABB_intersections.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_3/Robust_intersection_traits_3.h>

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/refine_mesh_3.h>

// IO
#include <CGAL/IO/Polyhedron_iostream.h>
//#include <CGAL/IO/Polyhedron_VRML_1_ostream.h>


//CGAL
struct K: public CGAL::Exact_predicates_inexact_constructions_kernel {};

using namespace sofa;

namespace cgal
{

template <class DataTypes>
MeshGenerationFromPolyhedron<DataTypes>::MeshGenerationFromPolyhedron()
    : f_X0( initData (&f_X0, "inputPoints", "Rest position coordinates of the degrees of freedom") )
    , f_triangles(initData(&f_triangles, "inputTriangles", "List of triangles"))
    , f_quads(initData(&f_quads, "inputQuads", "List of quads (if no triangles) "))
    , f_newX0( initData (&f_newX0, "outputPoints", "New Rest position coordinates from the tetrahedral generation") )
    , f_tetrahedra(initData(&f_tetrahedra, "outputTetras", "List of tetrahedra"))
    , facetAngle(initData(&facetAngle, 25.0, "facetAngle", "facetAngle"))
    , facetSize(initData(&facetSize, 0.15, "facetSize", "facetSize"))
    , facetApproximation(initData(&facetApproximation, 0.008, "facetApproximation", "facetApproximation"))
    , cellRatio(initData(&cellRatio, 4.0, "cellRatio", "cellRatio"))
    , cellSize(initData(&cellSize, 0.2, "cellSize", "cellSize"))
{

}

template <class DataTypes>
void MeshGenerationFromPolyhedron<DataTypes>::init()
{
    addInput(&f_X0);
    addInput(&f_triangles);
    addInput(&f_quads);

    addOutput(&f_newX0);
    addOutput(&f_tetrahedra);

    setDirtyValue();
}

template <class DataTypes>
void MeshGenerationFromPolyhedron<DataTypes>::reinit()
{

}

template <class DataTypes>
void MeshGenerationFromPolyhedron<DataTypes>::update()
{
    // Domain
    // (we use exact intersection computation with Robust_intersection_traits_3)

    typedef typename CGAL::Mesh_3::Robust_intersection_traits_3<K> Geom_traits;
    typedef typename CGAL::Polyhedron_3<Geom_traits> Polyhedron;
    typedef typename Polyhedron::HalfedgeDS HalfedgeDS;

    typedef typename CGAL::Polyhedral_mesh_domain_3<Polyhedron, Geom_traits> Mesh_domain;

    // Triangulation
    typedef typename CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
    typedef typename CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;

    // Mesh Criteria
    typedef typename CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
    typedef typename Mesh_criteria::Facet_criteria Facet_criteria;
    typedef typename Mesh_criteria::Cell_criteria Cell_criteria;

    typedef typename C3t3::Facet_iterator Facet_iterator;
    typedef typename C3t3::Cell_iterator Cell_iterator;

    typedef typename Tr::Finite_vertices_iterator Finite_vertices_iterator;
    typedef typename Tr::Vertex_handle Vertex_handle;
    typedef typename Tr::Point Point_3;

    const VecCoord& oldPoints = f_X0.getValue();
    const SeqTriangles& triangles = f_triangles.getValue();
    const SeqQuads& quads = f_quads.getValue();

    helper::WriteAccessor< Data<VecCoord> > newPoints = f_newX0;
    helper::WriteAccessor< Data<SeqTetrahedra> > tetrahedra = f_tetrahedra;

    if (triangles.empty() && quads.empty())
    {
        newPoints.clear();
        tetrahedra.clear();
        return;
    }
    if (!tetrahedra.empty()) return;

    // Create polyhedron
    Polyhedron polyhedron;
    AddTriangles<HalfedgeDS> builder(oldPoints, triangles, quads);
    polyhedron.delegate(builder);

//	std::ifstream input("share/mesh/elephant.off");
//        input >> polyhedron;

//    CGAL::set_ascii_mode( std::cout);
//    std::cout << "P : " << polyhedron.size_of_vertices() << std::endl;
//    std::cout << "F : " << polyhedron.size_of_facets() << std::endl;

    // Create domain
    Mesh_domain domain(polyhedron);

    // Set mesh criteria
    Facet_criteria facet_criteria(facetAngle.getValue(), facetSize.getValue(), facetApproximation.getValue()); // angle, size, approximation
    Cell_criteria cell_criteria(cellRatio.getValue(), cellSize.getValue()); // radius-edge ratio, size
    Mesh_criteria criteria(facet_criteria, cell_criteria);

    //Facet_criteria facet_criteria(25, 0.15, 0.008); // angle, size, approximation
    //Cell_criteria cell_criteria(4, 0.2); // radius-edge ratio, size
    //Mesh_criteria criteria(facet_criteria, cell_criteria);

    // Mesh generation
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);

    const Tr& tr = c3t3.triangulation();
    CGAL::Default_cell_index_pmap<C3t3> cell_pmap(c3t3);
    std::map<Vertex_handle, int> V;
    newPoints.clear();
    int inum = 0;
    for( Finite_vertices_iterator vit = tr.finite_vertices_begin(); vit != tr.finite_vertices_end(); ++vit)
    {
        V[vit] = inum++;
        Point_3 pointCgal = vit->point();
        Point p;
        p[0] = CGAL::to_double(pointCgal.x());
        p[1] = CGAL::to_double(pointCgal.y());
        p[2] = CGAL::to_double(pointCgal.z());

        newPoints.push_back(p);
    }

    tetrahedra.clear();
    for( Cell_iterator cit = c3t3.cells_begin() ; cit != c3t3.cells_end() ; ++cit )
    {
        //if (get(cell_pmap, cit)!=1) continue;
        Tetra tetra;
        for (int i=0; i<4; i++)
            tetra[i] = V[cit->vertex(i)];
        tetrahedra.push_back(tetra);
    }

}

} //cgal

#endif //CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_INL
