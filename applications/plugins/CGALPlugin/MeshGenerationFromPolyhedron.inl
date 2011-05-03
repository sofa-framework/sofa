/*
 * MeshGenerationFromPolyhedron.inl
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */

#ifndef CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_INL
#define CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_INL
#include "MeshGenerationFromPolyhedron.h"
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>

#include <CGAL/AABB_intersections.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_3/Robust_intersection_traits_3.h>

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/refine_mesh_3.h>
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,8,0)
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#endif

// IO
#include <CGAL/IO/Polyhedron_iostream.h>


//CGAL
//struct K: public CGAL::Exact_predicates_inexact_constructions_kernel {};
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

using namespace sofa;

//#ifdef SOFA_NEW_CGAL_MESH
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,5,0)
using namespace CGAL::parameters;
#endif

namespace cgal
{

template <class DataTypes>
MeshGenerationFromPolyhedron<DataTypes>::MeshGenerationFromPolyhedron()
    : f_X0( initData (&f_X0, "inputPoints", "Rest position coordinates of the degrees of freedom"))
    , f_triangles(initData(&f_triangles, "inputTriangles", "List of triangles"))
    , f_quads(initData(&f_quads, "inputQuads", "List of quads (if no triangles) "))
    , f_newX0( initData (&f_newX0, "outputPoints", "New Rest position coordinates from the tetrahedral generation"))
    , f_tetrahedra(initData(&f_tetrahedra, "outputTetras", "List of tetrahedra"))
    , frozen(initData(&frozen, false, "frozen", "true to prohibit recomputations of the mesh"))
    , facetAngle(initData(&facetAngle, 25.0, "facetAngle", "Lower bound for the angle in degrees of the surface mesh facets"))
    , facetSize(initData(&facetSize, 0.15, "facetSize", "Uniform upper bound for the radius of the surface Delaunay balls"))
    , facetApproximation(initData(&facetApproximation, 0.008, "facetApproximation", "Upper bound for the center-center distances of the surface mesh facets"))
    , cellRatio(initData(&cellRatio, 4.0, "cellRatio", "Upper bound for the radius-edge ratio of the tetrahedra"))
    , cellSize(initData(&cellSize, 0.2, "cellSize", "Uniform upper bound for the circumradii of the tetrahedra in the mesh"))
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,8,0)
    , sharpEdgeAngle(initData(&sharpEdgeAngle, 120.0, "sharpEdgeAngle", "Threshold angle to detect sharp edges in input surface (activated with CGAL 3.8+ if sharpEdgeSize > 0)"))
    , sharpEdgeSize(initData(&sharpEdgeSize, 0.0, "sharpEdgeSize", "Meshing size for sharp feature edges (activated with CGAL 3.8+ if sharpEdgeSize > 0)"))
#endif
    , odt(initData(&odt, false, "odt", "activate odt optimization"))
    , lloyd(initData(&lloyd, false, "lloyd", "activate lloyd optimization"))
    , perturb(initData(&perturb, false, "perturb", "activate perturb optimization"))
    , exude(initData(&exude, false, "exude", "activate exude optimization"))
    , odt_max_it(initData(&odt_max_it, 200, "odt_max_it", "odt max iteration number"))
    , lloyd_max_it(initData(&lloyd_max_it, 200, "lloyd_max_it", "lloyd max iteration number"))
    , perturb_max_time(initData(&perturb_max_time, 20.0, "perturb_max_time", "perturb maxtime"))
    , exude_max_time(initData(&exude_max_time, 20.0, "exude_max_time", "exude max time"))
    , drawTetras(initData(&drawTetras, false, "drawTetras", "display generated tetra mesh"))
    , drawSurface(initData(&drawSurface, false, "drawSurface", "display input surface mesh"))
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
    addInput(&frozen);
    addInput(&facetAngle);
    addInput(&facetSize);
    addInput(&facetApproximation);
    addInput(&cellRatio);
    addInput(&cellSize);
    addInput(&sharpEdgeAngle);
    addInput(&sharpEdgeSize);
    addInput(&odt);
    addInput(&lloyd);
    addInput(&perturb);
    addInput(&exude);
    addInput(&odt_max_it);
    addInput(&lloyd_max_it);
    addInput(&perturb_max_time);
    addInput(&exude_max_time);

    setDirtyValue();
}

template <class DataTypes>
void MeshGenerationFromPolyhedron<DataTypes>::reinit()
{
    sofa::core::DataEngine::reinit();
    update();
}

template <class C3t3>
int countWellCentered(C3t3& c3t3)
{
    int nb_in = 0;
    const typename C3t3::Triangulation& tri = c3t3.triangulation();
    for (typename C3t3::Cell_iterator cit = c3t3.cells_begin(); cit != c3t3.cells_end(); ++cit )
    {
        if (K().has_on_bounded_side_3_object()(tri.tetrahedron(cit),tri.dual(cit)))
        {
            ++nb_in;
        }
    }
    return nb_in;
}

template <class C3t3,class Obj>
void printStats(C3t3& c3t3, Obj* obj, const char* step = "")
{
    int nb_in = countWellCentered(c3t3);
    obj->sout << step << ":  number of tetra     = " << c3t3.number_of_cells() << obj->sendl;
    obj->sout << step << ":  well-centered tetra = " << ((double)nb_in/(double)c3t3.number_of_cells())*100 << "%" << obj->sendl;
}

template <class DataTypes>
void MeshGenerationFromPolyhedron<DataTypes>::update()
{
    // Domain
    // (we use exact intersection computation with Robust_intersection_traits_3)

#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,8,0)
    typedef typename CGAL::Mesh_3::Robust_intersection_traits_3<K> Geom_traits;
    //typedef K Geom_traits;
    typedef typename CGAL::Mesh_polyhedron_3<Geom_traits>::type Polyhedron;
    typedef typename Polyhedron::HalfedgeDS HalfedgeDS;
    typedef typename CGAL::Polyhedral_mesh_domain_with_features_3<Geom_traits, Polyhedron> Mesh_domain;
#else
    typedef typename CGAL::Mesh_3::Robust_intersection_traits_3<K> Geom_traits;
    typedef typename CGAL::Polyhedron_3<Geom_traits> Polyhedron;
    typedef typename Polyhedron::HalfedgeDS HalfedgeDS;
    typedef typename CGAL::Polyhedral_mesh_domain_3<Polyhedron, Geom_traits> Mesh_domain;
#endif


    // Triangulation
    typedef typename CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,8,0)
    typedef typename CGAL::Mesh_complex_3_in_triangulation_3<Tr, Mesh_domain::Corner_index, Mesh_domain::Curve_segment_index> C3t3;
#else
    typedef typename CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
#endif

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

    if (frozen.getValue()) return;
    newPoints.clear();
    tetrahedra.clear();
    if (triangles.empty() && quads.empty())
    {
        return;
    }
    //if (!tetrahedra.empty()) return;

    // Create polyhedron
    sout << "Create polyhedron" << sendl;
    Polyhedron polyhedron;
    AddTriangles<HalfedgeDS> builder(oldPoints, triangles, quads);
    polyhedron.delegate(builder);

//	std::ifstream input("share/mesh/elephant.off");
//        input >> polyhedron;

//    CGAL::set_ascii_mode( std::cout);
    sout << polyhedron.size_of_vertices() << " vertices, " << polyhedron.size_of_facets() << " facets." << sendl;

    if (polyhedron.size_of_vertices() == 0 || polyhedron.size_of_facets() == 0)
    {
        return;
    }
    // Create domain
    sout << "Create domain" << sendl;
    Mesh_domain domain(polyhedron);

#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,8,0)
    if (sharpEdgeSize.getValue() > 0)
    {
        sout << "Detect sharp edges (angle="<<sharpEdgeAngle.getValue()<<")" << sendl;
        domain.detect_features(sharpEdgeAngle.getValue());
    }
#endif

#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,6,0)
    // Mesh criteria (no cell_size set)
//    Mesh_criteria criteria(facet_angle=facetAngle.getValue(), facet_size=facetSize.getValue(), facet_distance=facetApproximation.getValue(),
//                           cell_radius_edge=cellRatio.getValue());
//    // Mesh generation
    sout << "Create Mesh" << sendl;
    Mesh_criteria criteria(
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,8,0)
        edge_size=sharpEdgeSize.getValue(),
#endif

        facet_angle=facetAngle.getValue(), facet_size=facetSize.getValue(), facet_distance=facetApproximation.getValue(),
        cell_radius_edge=cellRatio.getValue(), cell_size=cellSize.getValue());
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, no_perturb(), no_exude());

    // Set tetrahedron size (keep cell_radius_edge), ignore facets
//    Mesh_criteria new_criteria(cell_radius_edge=cellRatio.getValue(), cell_size=cellSize.getValue());

    // Mesh refinement
//	sout << "Refine Mesh" << sendl;
//    CGAL::refine_mesh_3(c3t3, domain, new_criteria);
#else
    // Set mesh criteria
    Facet_criteria facet_criteria(facetAngle.getValue(), facetSize.getValue(), facetApproximation.getValue()); // angle, size, approximation
    Cell_criteria cell_criteria(cellRatio.getValue(), cellSize.getValue()); // radius-edge ratio, size
    Mesh_criteria criteria(facet_criteria, cell_criteria);

    sout << "Create Mesh" << sendl;
    C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, no_perturb(), no_exude());
#endif
    printStats(c3t3,this,"Initial mesh");
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,5,0)
    sout << "Optimize Mesh" << sendl;
    if(lloyd.getValue())
    {
        CGAL::lloyd_optimize_mesh_3(c3t3, domain, max_iteration_number=lloyd_max_it.getValue());
        printStats(c3t3,this,"Lloyd");
    }
    if(odt.getValue())
    {
        CGAL::odt_optimize_mesh_3(c3t3, domain, max_iteration_number=odt_max_it.getValue());
        printStats(c3t3,this,"ODT");
    }
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(3,6,0)
    if(perturb.getValue())
    {
        CGAL::perturb_mesh_3(c3t3, domain, time_limit=perturb_max_time.getValue());
        printStats(c3t3,this,"Perturb");
    }
    if(exude.getValue())
    {
        CGAL::exude_mesh_3(c3t3, time_limit=exude_max_time.getValue());
        printStats(c3t3,this,"Exude");
    }
#else
    if(perturb.getValue())
    {
        CGAL::perturb_mesh_3(c3t3, domain, max_time=perturb_max_time.getValue());
        printStats(c3t3,this,"Perturb");
    }
    if(exude.getValue())
    {
        CGAL::exude_mesh_3(c3t3, max_time=exude_max_time.getValue());
        printStats(c3t3,this,"Exude");
    }
#endif
#endif

    const Tr& tr = c3t3.triangulation();

    std::map<Vertex_handle, int> Vnbe;

    for( Cell_iterator cit = c3t3.cells_begin() ; cit != c3t3.cells_end() ; ++cit )
    {
        for (int i=0; i<4; i++)
            ++Vnbe[cit->vertex(i)];
    }

    std::map<Vertex_handle, int> V;
    newPoints.clear();
    int inum = 0;
    int notconnected = 0;
    for( Finite_vertices_iterator vit = tr.finite_vertices_begin(); vit != tr.finite_vertices_end(); ++vit)
    {
        Point_3 pointCgal = vit->point();
        Point p;
        p[0] = CGAL::to_double(pointCgal.x());
        p[1] = CGAL::to_double(pointCgal.y());
        p[2] = CGAL::to_double(pointCgal.z());
        if (Vnbe.find(vit) == Vnbe.end() || Vnbe[vit] <= 0)
        {
            ++notconnected;
        }
        else
        {
            V[vit] = inum++;

            newPoints.push_back(p);
        }
    }
    if (notconnected > 0) serr << notconnected << " points are not connected to the mesh."<<sendl;

    tetrahedra.clear();
    for( Cell_iterator cit = c3t3.cells_begin() ; cit != c3t3.cells_end() ; ++cit )
    {
        Tetra tetra;
        for (int i=0; i<4; i++)
            tetra[i] = V[cit->vertex(i)];
        tetrahedra.push_back(tetra);
    }
    frozen.setValue(true);
}

template <class DataTypes>
void MeshGenerationFromPolyhedron<DataTypes>::draw()
{
    if (drawTetras.getValue())
    {
        helper::ReadAccessor< Data<VecCoord> > x = f_newX0;
        helper::ReadAccessor< Data<SeqTetrahedra> > tetrahedra = f_tetrahedra;

        //if (this->getContext()->getShowWireFrame())
        //    simulation::getSimulation()->DrawUtility().setPolygonMode(0,true);

        simulation::getSimulation()->DrawUtility().setLightingEnabled(false);
        std::vector< defaulttype::Vector3 > points[4];
        for(unsigned int i=0; i<tetrahedra.size(); ++i)
        {
            int a = tetrahedra[i][0];
            int b = tetrahedra[i][1];
            int c = tetrahedra[i][2];
            int d = tetrahedra[i][3];
            Coord center = (x[a]+x[b]+x[c]+x[d])*0.125;
            Coord pa = (x[a]+center)*(Real)0.666667;
            Coord pb = (x[b]+center)*(Real)0.666667;
            Coord pc = (x[c]+center)*(Real)0.666667;
            Coord pd = (x[d]+center)*(Real)0.666667;

// 		glColor4f(0,0,1,1);
            points[0].push_back(pa);
            points[0].push_back(pb);
            points[0].push_back(pc);

// 		glColor4f(0,0.5,1,1);
            points[1].push_back(pb);
            points[1].push_back(pc);
            points[1].push_back(pd);

// 		glColor4f(0,1,1,1);
            points[2].push_back(pc);
            points[2].push_back(pd);
            points[2].push_back(pa);

// 		glColor4f(0.5,1,1,1);
            points[3].push_back(pd);
            points[3].push_back(pa);
            points[3].push_back(pb);
        }

        simulation::getSimulation()->DrawUtility().drawTriangles(points[0], defaulttype::Vec<4,float>(0.0,0.0,1.0,1.0));
        simulation::getSimulation()->DrawUtility().drawTriangles(points[1], defaulttype::Vec<4,float>(0.0,0.5,1.0,1.0));
        simulation::getSimulation()->DrawUtility().drawTriangles(points[2], defaulttype::Vec<4,float>(0.0,1.0,1.0,1.0));
        simulation::getSimulation()->DrawUtility().drawTriangles(points[3], defaulttype::Vec<4,float>(0.5,1.0,1.0,1.0));

        //if (this->getContext()->getShowWireFrame())
        //    simulation::getSimulation()->DrawUtility().setPolygonMode(0,false);
    }
    if (drawSurface.getValue())
    {
        helper::ReadAccessor< Data<VecCoord> > x = f_X0;
        helper::ReadAccessor< Data<SeqTriangles> > triangles = f_triangles;
        helper::ReadAccessor< Data<SeqQuads> > quads = f_quads;

        if (this->getContext()->getShowWireFrame())
            simulation::getSimulation()->DrawUtility().setPolygonMode(0,true);

        simulation::getSimulation()->DrawUtility().setLightingEnabled(false);
        std::vector< defaulttype::Vector3 > points;
        for(unsigned int i=0; i<triangles.size(); ++i)
        {
            int a = triangles[i][0];
            int b = triangles[i][1];
            int c = triangles[i][2];
            Coord pa = x[a];
            Coord pb = x[b];
            Coord pc = x[c];
            points.push_back(pa);
            points.push_back(pb);
            points.push_back(pc);
        }
        for(unsigned int i=0; i<quads.size(); ++i)
        {
            int a = quads[i][0];
            int b = quads[i][1];
            int c = quads[i][2];
            int d = quads[i][3];
            Coord pa = x[a];
            Coord pb = x[b];
            Coord pc = x[c];
            Coord pd = x[d];
            points.push_back(pa);
            points.push_back(pb);
            points.push_back(pc);
            points.push_back(pa);
            points.push_back(pc);
            points.push_back(pd);
        }

        simulation::getSimulation()->DrawUtility().drawTriangles(points, defaulttype::Vec<4,float>(1.0,0.5,0.0,1.0));

        if (this->getContext()->getShowWireFrame())
            simulation::getSimulation()->DrawUtility().setPolygonMode(0,false);
    }
}

} //cgal

#endif //CGALPLUGIN_MESHGENERATIONFROMPOLYHEDRON_INL
