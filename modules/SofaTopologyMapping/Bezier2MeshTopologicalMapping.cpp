/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include <SofaTopologyMapping/Bezier2MeshTopologicalMapping.h>
#include <SofaGeneralTopology/BezierTetrahedronSetTopologyContainer.h>
#include <SofaGeneralTopology/BezierTriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
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

SOFA_DECL_CLASS ( Bezier2MeshTopologicalMapping )

// Register in the Factory
int Bezier2MeshTopologicalMappingClass = core::RegisterObject ( "This class maps a Bezier tetrahedral or triangular mesh into an interpolated refined tetrahedral or triangular mesh" )
        .add< Bezier2MeshTopologicalMapping >()
        ;

// Implementation
Bezier2MeshTopologicalMapping::Bezier2MeshTopologicalMapping ()
    : d_tesselationTetrahedronDegree ( initData ( &d_tesselationTetrahedronDegree, (unsigned int)0, "tesselatedTetrahedronDegree", "Tesselate a tetrahedral mesh as to create a Bezier Tetrahedral mesh of a given order" ) )
	, d_tesselationTriangleDegree ( initData ( &d_tesselationTriangleDegree, (unsigned int)0, "tesselatedTriangleDegree", "Tesselate a triangular  mesh as to create a Bezier Triangular mesh of a given order" ) )

{

}
Bezier2MeshTopologicalMapping::~Bezier2MeshTopologicalMapping(){
}

void Bezier2MeshTopologicalMapping::init()
{
//	initDone = true;
	if(!fromModel)
	{
		serr << "Could not find an input Bezier topology " << sendl;
		return;
	}
	if(!toModel)
	{
		serr << "Could not find an output mesh topology " << sendl;
		return;
	}
	toModel->clear();

	BezierTriangleSetTopologyContainer *btstc;
	fromModel->getContext()->get(btstc);

	if (!btstc) {
		serr << "Could not find an input  BezierTriangleSetTopologyContainer " <<sendl;
	}

	TriangleSetTopologyContainer *tstc;
	toModel->getContext()->get(tstc);

	if (!tstc) {
		serr << "Could not find an output  TriangleSetTopologyContainer " <<sendl;
	}

	size_t bezierTesselation=d_tesselationTriangleDegree.getValue();
	// by default use the degree of the bezier triangles
	if (bezierTesselation==0)
		bezierTesselation=btstc->getDegree();

	PointSetTopologyModifier *toPointMod = NULL;
	toModel->getContext()->get(toPointMod, sofa::core::objectmodel::BaseContext::Local);
	EdgeSetTopologyModifier *toEdgeMod = NULL;
	toModel->getContext()->get(toEdgeMod, sofa::core::objectmodel::BaseContext::Local);
	TriangleSetTopologyModifier *toTriangleMod = NULL;
	toModel->getContext()->get(toTriangleMod, sofa::core::objectmodel::BaseContext::Local);
//	TetrahedronSetTopologyModifier *toTetrahedronMod = NULL;
//	toModel->getContext()->get(toTetrahedronMod, sofa::core::objectmodel::BaseContext::Local);


	if (bezierTesselation>0) {
		 nbPoints=btstc->getNumberOfTriangularPoints()+btstc->getNumberOfEdges()*(bezierTesselation-1)+
			btstc->getNumberOfTriangles()*(bezierTesselation-1)*(bezierTesselation-2)/2;

		// fill topology container with empty points instead of specifying its size
		// the real position will be filled by the mechanical mapping
		 size_t i;
		tstc->setNbPoints((int)nbPoints);
		// for (i=0;i<nbPoints;++i)
		//	 tstc->addPoint((SReal)0.0f,(SReal)0.0f,(SReal)0.0f);

		 // store the vertices of the macro triangles since the set of positions may be a large overset of the positions of the triangular bezier triangles (if a Tetra2Trian mapping is used)
		 std::set<size_t> triangleVertexSet;
		

         const core::topology::BaseMeshTopology::SeqTriangles &ta=btstc->getTriangleArray();

		  for (i=0;i<ta.size();++i)  {
              core::topology::BaseMeshTopology::Triangle t=ta[i];
			  triangleVertexSet.insert(t[0]);triangleVertexSet.insert(t[1]);triangleVertexSet.insert(t[2]);
		  }
		  assert(btstc->getNumberOfTriangularPoints()==triangleVertexSet.size());

		  global2LocalBezierVertexArray.resize(btstc->getNbPoints());
		  std::fill(global2LocalBezierVertexArray.begin(),global2LocalBezierVertexArray.end(),-1);

		  std::set<size_t>::iterator itvs;
		  for (i=0,itvs=triangleVertexSet.begin();itvs!=triangleVertexSet.end();++itvs,++i) {
			  local2GlobalBezierVertexArray.push_back(*itvs);
			  global2LocalBezierVertexArray[*itvs]=i;
		  }
		/*
		   if (toPointMod)
    {
        toPointMod->addPointsProcess(pBaryCoords.size());
    }
    else
    {
        for (unsigned int j = 0; j < pBaryCoords.size(); j++)
        {        
            toModel->addPoint(fromModel->getPX(i) + pBaryCoords[j][0], fromModel->getPY(i) + pBaryCoords[j][1], fromModel->getPZ(i) + pBaryCoords[j][2]);
        }
    }
    
    for (unsigned int j = 0; j < pBaryCoords.size(); j++)
    {        
        pointsMappedFrom[POINT][i].push_back(pointSource.size());
        pointSource.push_back(std::make_pair(POINT, i));
    }
    
    if (toPointMod)
    {  
        helper::vector< helper::vector< unsigned int > > ancestors;
        helper::vector< helper::vector< double       > > coefs;
        toPointMod->addPointsWarning(pBaryCoords.size(), ancestors, coefs);
    }
	*/

		 // this is used to store the global indices of tessellated Bezier triangles
		 globalIndexTesselatedBezierTriangleArray.resize(btstc->getNumberOfTriangles());
		 // get the division of the triangle into subtriangles
		 sofa::helper::vector< sofa::component::topology::BezierTriangleSetTopologyContainer::LocalTriangleIndex> 
			 sta=btstc->getLocalIndexSubtriangleArrayOfGivenDegree(bezierTesselation);

		// handle triangles
		size_t nbTriangles=btstc->getNumberOfTriangles()*(bezierTesselation*bezierTesselation);

		helper::WriteOnlyAccessor<Data<sofa::helper::vector <unsigned int>  > > loc2glob=Loc2GlobDataVec;
		loc2glob.resize(nbTriangles);

		size_t j,k,l,offset,rank;
		size_t baseEdgeOffset=btstc->getNumberOfTriangularPoints();
		size_t baseTriangleOffset=baseEdgeOffset+btstc->getNumberOfEdges()*(bezierTesselation-1);
		size_t pointsPerTriangle=(bezierTesselation-1)*(bezierTesselation-2)/2;
        core::topology::BaseMeshTopology::Triangle subtriangle;
		sofa::component::topology::BezierTriangleSetTopologyContainer::VecPointID indexArray;
		sofa::helper::vector<size_t>  bezierEdge;
		sofa::component::topology::TriangleBezierIndex trbi;
		size_t bezierDegree=btstc->getDegree();

		for (rank=0,i=0;i<btstc->getNumberOfTriangles();++i) {
			// there are (bezierTesselation)*(bezierTesselation) subtriangles
			// first store the indices of all the macro triangles into an array
            core::topology::BaseMeshTopology::Triangle tr=btstc->getTriangle(i);
			sofa::helper::vector<size_t> macroTriangleIndexArray;
			// store the 3 vertices

			for (j=0;j<3;++j) {
				macroTriangleIndexArray.push_back(global2LocalBezierVertexArray[tr[j]]);
			}
//			std::cerr << std::endl;
	//		for(j=0;j<tesselatedTriangleIndices.size();++j)
		//		std::cerr<< (sofa::defaulttype::Vec<3,size_t >)(tesselatedTriangleIndices[j])<<std::endl;

			// store the edge point
			if (bezierTesselation>1) {
                core::topology::BaseMeshTopology::EdgesInTriangle eit=btstc->getEdgesInTriangle(i);
				 btstc->getGlobalIndexArrayOfBezierPointsInTriangle(i, indexArray);
				for (j=0;j<3;++j) {

					// store the edge in an array only once 
                    const core::topology::BaseMeshTopology::TrianglesAroundEdge &tae=  btstc->getTrianglesAroundEdge(eit[j]);
					if (tae[0]==i) {
                        const core::topology::BaseMeshTopology::Edge &e= btstc->getEdge(eit[j]);
						edgeTriangleArray.push_back(e);
						// find the edge index
						for(k=0;tr[k]==e[0]||tr[k]==e[1];++k);
						assert(tr[(k+1)%3]==e[0]);
						bezierEdge.clear();
						for(l=1;l<=(bezierDegree-1);++l) {

							trbi[k]=0;
							trbi[(k+1)%3]=bezierDegree-l;
							trbi[(k+2)%3]=l;

							bezierEdge.push_back(indexArray[  btstc->getLocalIndexFromTriangleBezierIndex(trbi)]);
						}						  
						bezierEdgeArray.push_back(bezierEdge);
					}




                    core::topology::BaseMeshTopology::Edge e=btstc->getEdge(eit[j]);
					offset=baseEdgeOffset+eit[j]*(bezierTesselation-1);
					if (e[0]==tr[(j+1)%3] ) {
						for (k=0;k<(size_t)(bezierTesselation-1);++k) {
							macroTriangleIndexArray.push_back(offset+k);
						}
					} else {
						for (k=bezierTesselation-1;k>0;--k) {
							macroTriangleIndexArray.push_back(offset+k-1);
						}
					}

				}
			}
			// store the triangle point
			if (bezierTesselation>2) {
				offset=baseTriangleOffset+i*pointsPerTriangle;
				for (j=0;j<pointsPerTriangle;++j) {
					macroTriangleIndexArray.push_back(offset+j);
				}
			}
			// save the global indices of each tessellated triangle for the computation of normals
			globalIndexTesselatedBezierTriangleArray[i]=macroTriangleIndexArray;
	
//			std::cerr<< macroTriangleIndexArray<<std::endl;
			// now add subtriangles to the list
			for (j=0;j<sta.size();++j,++rank){
				for(k=0;k<3;++k){
					subtriangle[k]=macroTriangleIndexArray[sta[j][k]];
				}
				// add the subtriangle
				if (toTriangleMod)
					toTriangleMod->addTriangleProcess(subtriangle);
				else
					tstc->addTriangle(subtriangle[0],subtriangle[1],subtriangle[2]);
				// update the topological maps
				loc2glob[rank]=i;
				Glob2LocMap.insert(std::pair<unsigned int, unsigned int>(i,rank));
			}
		  }
	}
   
}

void Bezier2MeshTopologicalMapping::updateTopologicalMappingTopDown() {
	// TO DO : Handle topological changes
}


} // namespace topology
} // namespace component
} // namespace sofa

