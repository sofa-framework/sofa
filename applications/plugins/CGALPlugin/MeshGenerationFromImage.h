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
#ifndef CGALPLUGIN_MESHGENERATIONFROMIMAGE_H
#define CGALPLUGIN_MESHGENERATIONFROMIMAGE_H

#define CGAL_MESH_3_VERBOSE

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/core/visual/VisualParams.h>

#include <CGAL/version.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Labeled_image_mesh_domain_3.h>
#include <CGAL/Mesh_domain_with_polyline_features_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/refine_mesh_3.h>
#include <CGAL/Image_3.h>
#include <CGAL/Weighted_point.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/rmath.h>
#include <image/ImageTypes.h>

//CGAL
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

namespace cgal
{

using sofa::helper::vector;
using cimg_library::CImg;

template <class DataTypes, class _ImageTypes>
class MeshGenerationFromImage : public sofa::core::DataEngine
{

public:
    typedef sofa::core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE2(MeshGenerationFromImage,DataTypes,_ImageTypes),Inherited);

    typedef typename sofa::defaulttype::Vec3dTypes::Real Real;
    typedef typename sofa::defaulttype::Vec3dTypes::Coord Point;
    typedef typename sofa::defaulttype::Vec3dTypes::Coord Coord;
    typedef typename sofa::defaulttype::Vec3dTypes::VecCoord VecCoord;
    typedef sofa::defaulttype::Vector3 Vector3;

    typedef sofa::core::topology::BaseMeshTopology::Tetra Tetra;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;

	// Domain
    // (we use exact intersection computation with Robust_intersection_traits_3)
    typedef CGAL::Mesh_domain_with_polyline_features_3<CGAL::Labeled_image_mesh_domain_3<CGAL::Image_3,K> >    Mesh_domain;
    typedef K::Point_3 Point3;

    typedef std::vector<Point3>	 Polyline;
    typedef std::list<Polyline>	 Polylines;

    // Triangulation
    typedef typename CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
    typedef typename CGAL::Mesh_complex_3_in_triangulation_3<Tr,Mesh_domain::Corner_index,Mesh_domain::Curve_segment_index> C3t3;

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

    // image data
    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef sofa::helper::WriteAccessor<sofa::core::objectmodel::Data< ImageTypes > > waImage;
    typedef sofa::helper::ReadAccessor<sofa::core::objectmodel::Data< ImageTypes > > raImage;

    // transform data
    typedef SReal t_Real;
    typedef sofa::defaulttype::ImageLPTransform<t_Real> TransformType;
    typedef sofa::helper::WriteAccessor<sofa::core::objectmodel::Data< TransformType > > waTransform;
    typedef sofa::helper::ReadAccessor<sofa::core::objectmodel::Data< TransformType > > raTransform;


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

    static std::string templateName(const MeshGenerationFromImage<DataTypes, _ImageTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //Inputs    
    sofa::core::objectmodel::DataFileName d_filename;
    sofa::core::objectmodel::Data< ImageTypes > d_image;
    sofa::core::objectmodel::Data< TransformType > d_transform;
    sofa::core::objectmodel::Data<VecCoord> d_features;

    //Outputs
    sofa::core::objectmodel::Data<VecCoord> d_newX0;
    sofa::core::objectmodel::Data<SeqTetrahedra> d_tetrahedra;
    sofa::core::objectmodel::Data<sofa::helper::vector<int> > d_tetraDomain;
    sofa::core::objectmodel::Data<sofa::helper::vector<double> > d_outputCellData;
    sofa::core::objectmodel::Data<bool> d_frozen;

    //Parameters
    sofa::core::objectmodel::Data<double> d_edgeSize, d_facetAngle, d_facetSize, d_facetApproximation;
    sofa::core::objectmodel::Data<double> d_cellRatio;
    sofa::core::objectmodel::Data<double> d_cellSize;
    sofa::core::objectmodel::Data< sofa::helper::vector<int> > d_label;
    sofa::core::objectmodel::Data< sofa::helper::vector<double> > d_labelCellSize;
    sofa::core::objectmodel::Data< sofa::helper::vector<double> > d_labelCellData;
    sofa::core::objectmodel::Data<bool> d_odt, d_lloyd, d_perturb, d_exude;
    sofa::core::objectmodel::Data<int> d_odtMaxIt, d_lloydMaxIt;
    sofa::core::objectmodel::Data<double> d_perturbMaxTime, d_exudeMaxTime;
    sofa::core::objectmodel::Data<int> d_ordering;

    // Display
    sofa::core::objectmodel::Data<bool> d_drawTetras;

    sofa::helper::vector<int> m_tetraDomainLabels;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(CGALPLUGIN_MESHGENERATIONFROMIMAGE_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CGALPLUGIN_API MeshGenerationFromImage<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ImageUC>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_CGALPLUGIN_API MeshGenerationFromImage<sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ImageUC>;
#endif //SOFA_DOUBLE
#endif

} //cgal

#endif /* CGALPLUGIN_MESHGENERATIONFROMIMAGE_H */
