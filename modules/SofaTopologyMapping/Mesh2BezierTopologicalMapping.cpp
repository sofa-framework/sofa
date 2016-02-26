/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaTopologyMapping/Mesh2BezierTopologicalMapping.h>
#include <SofaBaseTopology/BezierTetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/BezierTriangleSetTopologyContainer.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>


#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{
namespace component
{
namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::component::topology;
using namespace sofa::core::topology;

SOFA_DECL_CLASS ( Mesh2BezierTopologicalMapping )

// Register in the Factory
int Mesh2BezierTopologicalMappingClass = core::RegisterObject ( "This class maps a tetrahedral mesh mesh into a high order tetrahedral mesh" )
        .add< Mesh2BezierTopologicalMapping >()
        ;

// Implementation
Mesh2BezierTopologicalMapping::Mesh2BezierTopologicalMapping ()
    : bezierTetrahedronDegree ( initData ( &bezierTetrahedronDegree, (unsigned int)0, "bezierTetrahedronDegree", "Tesselate a tetrahedral mesh as to create a Bezier Tetrahedral mesh of a given order" ) )
	, bezierTriangleDegree ( initData ( &bezierTriangleDegree, (unsigned int)0, "bezierTriangleDegree", "Tesselate a triangular  mesh as to create a Bezier Triangular mesh of a given order" ) )

{

}

void Mesh2BezierTopologicalMapping::init()
{
   
	if (bezierTetrahedronDegree.getValue()>0) {
		size_t degree=bezierTetrahedronDegree.getValue();
		// make copyTetrahedra as true
		copyTetrahedra.setValue(true);

		// process each coord array
		helper::WriteAccessor< Data<vector< Vec3d > > > pBary = pointBaryCoords;
		pBary.clear();
		pBary.push_back(Vec3d(0,0,0));
		if (degree >1) {
			// process each edge array
			helper::WriteAccessor< Data<vector< Vec3d > > > eBary = edgeBaryCoords;
			eBary.clear();
			size_t i;
			for (i=1;i<degree;++i) {
				eBary.push_back(Vec3d((double)i/degree,(double)(degree-i)/degree,0));
			}
			
		}
		if (degree >2) {
			// process each triangle array
			helper::WriteAccessor< Data<vector< Vec3d > > > trBary = triangleBaryCoords;
			trBary.clear();
			size_t i,j;
			for (i=1;i<(degree-1);++i) {
				for (j=1;j<(degree-i);++j) {
					trBary.push_back(Vec3d((double)j/degree,(double)(degree-i-j)/degree,(double)i/degree));
//					trBary.push_back(Vec3d((double)(i)/degree,(double)j/degree,(double)(degree-i-j)/degree));
				}
			}
			
		}
		if (degree >3) {
			// process each tetrahedron array
			helper::WriteAccessor< Data<vector< Vec3d > > > tetBary = tetraBaryCoords;
			tetBary.clear();
			size_t i,j,k;
			for (i=1;i<(degree-2);++i) {
				for (j=1;j<(degree-i-1);++j) {
					for (k=1;k<(degree-j-i);++k) {
						tetBary.push_back(Vec3d((double)j/degree,(double)k/degree,(double)(degree-i-j-k)/degree));
//						tetBary.push_back(Vec3d((double)i/degree,(double)j/degree,(double)k/degree));
					}
				}
			}
			
		}

	} 

	if (bezierTriangleDegree.getValue()>0) {
		size_t degree=bezierTriangleDegree.getValue();
		// make copytriangles as true
		copyTriangles.setValue(true);

		// process each coord array
		helper::WriteAccessor< Data<vector< Vec3d > > > pBary = pointBaryCoords;
		pBary.clear();
		pBary.push_back(Vec3d(0,0,0));
		if (degree >1) {
			// process each edge array
			helper::WriteAccessor< Data<vector< Vec3d > > > eBary = edgeBaryCoords;
			eBary.clear();
			size_t i;
			for (i=1;i<degree;++i) {
				eBary.push_back(Vec3d((double)i/degree,(double)(degree-i)/degree,0));
			}

		}
		if (degree >2) {
			// process each triangle array
			helper::WriteAccessor< Data<vector< Vec3d > > > trBary = triangleBaryCoords;
			trBary.clear();
			size_t i,j;
			for (i=1;i<(degree-1);++i) {
				for (j=1;j<(degree-i);++j) {
					trBary.push_back(Vec3d((double)j/degree,(double)(degree-i-j)/degree,(double)i/degree));
					//					trBary.push_back(Vec3d((double)(i)/degree,(double)j/degree,(double)(degree-i-j)/degree));
				}
			}

		}
	} 
	Mesh2PointTopologicalMapping::init();
	if (bezierTetrahedronDegree.getValue()>0) {
		/// copy the number of tetrahedron vertices to the Bezier topology container
		BezierTetrahedronSetTopologyContainer *toBTTC = NULL;
		toModel->getContext()->get(toBTTC, sofa::core::objectmodel::BaseContext::Local);
		if (toBTTC) {
			// set the number of tetrahedral vertices among the number of control points.
			toBTTC->d_numberOfTetrahedralPoints.setValue(pointsMappedFrom[POINT].size());
			toBTTC->d_degree.setValue(bezierTetrahedronDegree.getValue());
			toBTTC->reinit();
			toBTTC->checkBezierPointTopology();
		} else {
			serr << "Could not find a BezierTetrahedronSetTopologyContainer as target topology " << sendl;
		}
	}
	if (bezierTriangleDegree.getValue()>0) {
		/// copy the number of triangle vertices to the Bezier triangle topology container
		BezierTriangleSetTopologyContainer *toBTTC = NULL;
		toModel->getContext()->get(toBTTC, sofa::core::objectmodel::BaseContext::Local);
		if (toBTTC) {
			// set the number of tetrahedral vertices among the number of control points.
			toBTTC->d_numberOfTriangularPoints.setValue(pointsMappedFrom[POINT].size());
			toBTTC->d_degree.setValue(bezierTriangleDegree.getValue());
			toBTTC->reinit();
			toBTTC->checkBezierPointTopology();
		} else {
			serr << "Could not find a BezierTriangleSetTopologyContainer as target topology " << sendl;
		}
	}
   
}


} // namespace topology
} // namespace component
} // namespace sofa

