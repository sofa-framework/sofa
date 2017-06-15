/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <CGAL/make_mesh_3.h>
#include <CGAL/refine_mesh_3.h>

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

    //SOFA_CLASS(SOFA_TEMPLATE(MeshGenerationFromImage,DataTypes),sofa::core::DataEngine);

    typedef typename sofa::defaulttype::Vec3dTypes::Real Real;
    typedef typename sofa::defaulttype::Vec3dTypes::Coord Point;
    typedef typename sofa::defaulttype::Vec3dTypes::Coord Coord;
    typedef typename sofa::defaulttype::Vec3dTypes::VecCoord VecCoord;
    typedef sofa::defaulttype::Vector3 Vector3;

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
    sofa::core::objectmodel::DataFileName m_filename;
    sofa::core::objectmodel::Data< ImageTypes > image;
    sofa::core::objectmodel::Data< TransformType > transform;

    //Outputs
    sofa::core::objectmodel::Data<VecCoord> f_newX0;
    sofa::core::objectmodel::Data<SeqTetrahedra> f_tetrahedra;
    sofa::core::objectmodel::Data<sofa::helper::vector<int> > f_tetraDomain;
    vector<int> tetraDomainLabels;
    sofa::core::objectmodel::Data<sofa::helper::vector<double> > output_cellData;

    sofa::core::objectmodel::Data<bool> frozen;

    //Parameters
    sofa::core::objectmodel::Data<double> facetAngle, facetSize, facetApproximation;
    sofa::core::objectmodel::Data<double> cellRatio;
    sofa::core::objectmodel::Data<double> cellSize;
    sofa::core::objectmodel::Data< sofa::helper::vector<int> > label;
    sofa::core::objectmodel::Data< sofa::helper::vector<double> > labelCellSize;
    sofa::core::objectmodel::Data< sofa::helper::vector<double> > labelCellData;
    sofa::core::objectmodel::Data<bool> odt, lloyd, perturb, exude;
    sofa::core::objectmodel::Data<int> odt_max_it, lloyd_max_it;
    sofa::core::objectmodel::Data<double> perturb_max_time, exude_max_time;
    sofa::core::objectmodel::Data<int> ordering;

    // Display
    sofa::core::objectmodel::Data<bool> drawTetras;
    sofa::core::objectmodel::Data<bool> drawSurface;

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
