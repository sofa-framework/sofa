/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef CGALPLUGIN_MESHGENERATIONFROMIMAGE_H
#define CGALPLUGIN_MESHGENERATIONFROMIMAGE_H

#define CGAL_MESH_3_VERBOSE

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/visual/VisualParams.h>

#include <CGAL/version.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Labeled_image_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/refine_mesh_3.h>

//CGAL
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

namespace cgal
{

template <class DataTypes>
class MeshGenerationFromImage : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MeshGenerationFromImage,DataTypes),sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef sofa::core::topology::BaseMeshTopology::Tetra Tetra;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;

	// Domain
    // (we use exact intersection computation with Robust_intersection_traits_3)
    typedef typename CGAL::Labeled_image_mesh_domain_3<CGAL::Image_3,K> Mesh_domain;

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
	typedef CGAL::Mesh_constant_domain_field_3<Mesh_domain::R,
                                           Mesh_domain::Index> Sizing_field;


public:
    MeshGenerationFromImage();
    virtual ~MeshGenerationFromImage() { }

    void init();
    void reinit();

    void update();

    void draw(const sofa::core::visual::VisualParams* vparams);

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const MeshGenerationFromImage<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Inputs
    sofa::core::objectmodel::DataFileName m_filename;

    //Outputs
    Data<VecCoord> f_newX0;
    Data<SeqTetrahedra> f_tetrahedra;

    Data<bool> frozen;

    //Parameters
    Data<double> facetAngle, facetSize, facetApproximation;
    Data<double> cellRatio;
	Data<double> cellSize;
	Data< sofa::helper::vector<int> > label;
	Data< sofa::helper::vector<double> > labelCellSize;
    Data<bool> odt, lloyd, perturb, exude;
    Data<int> odt_max_it, lloyd_max_it;
    Data<double> perturb_max_time, exude_max_time;
    Data<int> ordering;

    // Display
    Data<bool> drawTetras;
    Data<bool> drawSurface;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(CGALPLUGIN_MESHGENERATIONFROMIMAGE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CGALPLUGIN_API MeshGenerationFromImage<sofa::defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_CGALPLUGIN_API MeshGenerationFromImage<sofa::defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} //cgal

#endif /* CGALPLUGIN_MESHGENERATIONFROMIMAGE_H */
