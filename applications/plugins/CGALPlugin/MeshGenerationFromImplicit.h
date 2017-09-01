#ifndef MESH_GENERATION_FROM_DG_H
#define MESH_GENERATION_FROM_DG_H

//Mesh Volume
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Implicit_to_labeling_function_wrapper.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Bbox_3.h>

//Topology
#include <SofaBaseTopology/MeshTopology.h>

//Shapes
#include <SofaVolumetricData/DistanceGrid.h>
#include <SofaVolumetricData/DistanceGridComponent.h>
#include <SofaVolumetricData/ImplicitSphere.h>

//Misc
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/objectmodel/Link.h>
#include <fstream>


typedef sofa::component::container::DistanceGrid DistanceGrid;
typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
typedef sofa::core::topology::BaseMeshTopology::Tetra Tetra;
typedef sofa::defaulttype::Vector3 Coord;
typedef sofa::helper::vector<Coord> VecCoord;

using namespace sofa::core;
using namespace sofa::core::visual;
using namespace sofa::core::objectmodel;

namespace cgal {

class MeshGenerationFromImplicitShape : public BaseObject {

public:
    SOFA_CLASS(MeshGenerationFromImplicitShape, BaseObject);
    void draw(const VisualParams* vparams);
    MeshGenerationFromImplicitShape()
        : in_facetsize(initData(&in_facetsize,"facet_size","size of facet"))
        , in_approximation(initData(&in_approximation,"approximation","approximation"))
        , in_cellsize(initData(&in_cellsize,"cell_size","size of cell"))
        , xmin_box(initData(&xmin_box,0.0,"xmin_box","xmin of bbox"))
        , ymin_box(initData(&ymin_box,0.0,"ymin_box","ymin of bbox"))
        , zmin_box(initData(&zmin_box,-5.0,"zmin_box","zmin of bbox"))
        , xmax_box(initData(&xmax_box,27.0,"xmax_box","xmax of bbox"))
        , ymax_box(initData(&ymax_box,27.0,"ymax_box","ymax of bbox"))
        , zmax_box(initData(&zmax_box,5.0,"zmax_box","zmax of bbox"))
        , drawTetras(initData(&drawTetras,false,"drawTetras","display generated tetra mesh"))
        , out_Points(initData(&out_Points, "outputPoints", "position coordinates from the tetrahedral generation"))
        , out_tetrahedra(initData(&out_tetrahedra, "outputTetras", "list of tetrahedra"))
        , in_grid(initLink("grid", "Grid"))
        , in_function(initLink("function", "Function"))
    {
    }
    virtual ~MeshGenerationFromImplicitShape() { }
    int volumeMeshGeneration(float facet_size, float approximation, float cell_size);
    virtual void init();

private:
    CGAL::Bbox_3 BoundingBox(double x_min, double y_min, double z_min, double x_max, double y_max, double z_max);
    //Inputs and atritbutes
    Data<float> in_facetsize;
    Data<float> in_approximation;
    Data<float> in_cellsize;
    Data<double> xmin_box, ymin_box, zmin_box, xmax_box, ymax_box, zmax_box;
    //Display
    Data<bool> drawTetras;
    //Output
    Data<VecCoord> out_Points;
    Data<SeqTetrahedra> out_tetrahedra;
    //Link
    typedef SingleLink< MeshGenerationFromImplicitShape, DistanceGridComponent, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkGrid;
    LinkGrid in_grid;
    typedef SingleLink< MeshGenerationFromImplicitShape, ImplicitSphere, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkFunction;
    LinkFunction in_function;

};


} //namespace cgal

#endif //MESH_GENERATION_FROM_DG_H
