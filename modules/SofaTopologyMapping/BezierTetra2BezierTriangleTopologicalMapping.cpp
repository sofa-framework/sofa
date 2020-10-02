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
#include "config.h"
#include <SofaTopologyMapping/BezierTetra2BezierTriangleTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaGeneralTopology/BezierTriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <SofaGeneralTopology/BezierTetrahedronSetTopologyContainer.h>
#include <sofa/core/topology/TopologyChange.h>

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

SOFA_DECL_CLASS(BezierTetra2BezierTriangleTopologicalMapping)

// Register in the Factory
int BezierTetra2BezierTriangleTopologicalMappingClass = core::RegisterObject("Special case of mapping where BezierTetrahedronSetTopology is converted to BezierTriangleSetTopology")
        .add< BezierTetra2BezierTriangleTopologicalMapping >()

        ;

// Implementation

BezierTetra2BezierTriangleTopologicalMapping::BezierTetra2BezierTriangleTopologicalMapping()
    : flipNormals(initData(&flipNormals, bool(false), "flipNormals", "Flip Normal ? (Inverse point order when creating triangle)"))
  {
}

BezierTetra2BezierTriangleTopologicalMapping::~BezierTetra2BezierTriangleTopologicalMapping()
{
}

void BezierTetra2BezierTriangleTopologicalMapping::init()
{
    //sout << "INFO_print : init BezierTetra2BezierTriangleTopologicalMapping" << sendl;

    // INITIALISATION of Bezier TRIANGULAR mesh from Bezier TETRAHEDRAL mesh :


    if (fromModel)
    {
		BezierTetrahedronSetTopologyContainer *from_btstc;
		fromModel->getContext()->get(from_btstc);
		if (!from_btstc) {
			serr << "Could not find an input BezierTetrahedronSetTopologyContainer"<<sendl;
		}



        if (toModel)
        {

//            sout << "INFO_print : BezierTetra2BezierTriangleTopologicalMapping - to = triangle" << sendl;

            BezierTriangleSetTopologyContainer *to_btstc;
            toModel->getContext()->get(to_btstc);

			if (!to_btstc) {
				serr << "Could not find an output  BezierTriangleSetTopologyContainer " <<sendl;
			}

            to_btstc->clear();

			// set the number of points of Bezier triangle = number of points of Bezier tetra
            toModel->setNbPoints(from_btstc->getNbPoints());

           TriangleSetTopologyModifier *to_tstm;
            toModel->getContext()->get(to_tstm);

            const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle> &triangleArray=fromModel->getTriangles();
            const bool flipN = flipNormals.getValue();

			// set the degree of Bezier triangle equal to that of Bezier tetra
			to_btstc->d_degree.setValue(from_btstc->getDegree());
			to_btstc->init();

			// initialize table of equivalence
			sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

			Loc2GlobVec.clear();
			Glob2LocMap.clear();
			size_t rankTriangle=0;
			// set to count the number of vertices 
			std::set<size_t> vertexSet;
			// set the boolean indicating if the triangulation is rational
			helper::WriteOnlyAccessor<Data <BezierTriangleSetTopologyContainer::SeqBools> >  isRationalSpline=to_btstc->d_isRationalSpline;

			for (unsigned int i=0; i<triangleArray.size(); ++i)
			{
				/// find triangles on the border of the tetrahedral mesh 
                core::topology::BaseMeshTopology::TetrahedraAroundTriangle tat=fromModel->getTetrahedraAroundTriangle(i);
				if (tat.size()==1)
				{
					// add macro triangle
                    core::topology::BaseMeshTopology::Triangle t = triangleArray[i];
					if(flipN)
					{
						unsigned int tmp = t[2];
						t[2] = t[1];
						t[1] = tmp;
					}
					to_tstm->addTriangleProcess(t);
					// add vertices in set
					vertexSet.insert(t[0]);vertexSet.insert(t[1]);vertexSet.insert(t[2]);
					//  if the adjacent tetrahedron is rational then the triangle is also rational
					const bool irs=from_btstc->isRationalSpline(tat[0]);
					isRationalSpline.push_back(irs);

					Loc2GlobVec.push_back(i);
					Glob2LocMap[i]=Loc2GlobVec.size()-1;
					// update the local maps of control points
					// get the control points of the tetrahedron adjacent to that triangle
					const BezierTriangleSetTopologyContainer::VecPointID  &indexArray=
						from_btstc->getGlobalIndexArrayOfBezierPoints(tat[0]);
                    core::topology::BaseMeshTopology::Tetrahedron tet=from_btstc->getTetrahedron(tat[0]);
					size_t k,j,l,equiv[3];
					// get the index of that triangle in the tetrahedron
                    core::topology::BaseMeshTopology::TrianglesInTetrahedron tit=from_btstc->getTrianglesInTetrahedron(tat[0]);
					for (l=0;tit[l]!=i;++l);
					// find the equivalence between the triangle index and tetrahedron index
					for(j=0;j<3;++j) {
						for(k=0;tet[k]!=t[j];++k);
						equiv[j]=k;
					}
					// now gets the indices of all control points in the Bezier triangle
					sofa::helper::vector<TriangleBezierIndex> trbia=to_btstc->getTriangleBezierIndexArray();
					for (j=0;j<trbia.size();j++) {
						// finds the TetrahedronBezierIndex tbi associated with the TriangleBezierIndex trbia[j]
						TetrahedronBezierIndex tbi;
						tbi[l]=0;
						for(k=0;k<3;++k) {
							tbi[equiv[k]]= trbia[j][k];
						}
						size_t globalIndex=indexArray[from_btstc->getLocalIndexFromTetrahedronBezierIndex(tbi)];
						// now fill the Bezier triangle maps
						to_btstc->locationToGlobalIndexMap.insert(std::pair<BezierTriangleSetTopologyContainer::ControlPointLocation,size_t>(BezierTriangleSetTopologyContainer::ControlPointLocation(rankTriangle,trbia[j]),globalIndex));
						to_btstc->globalIndexToLocationMap.insert(std::pair<size_t,BezierTriangleSetTopologyContainer::ControlPointLocation>(globalIndex,BezierTriangleSetTopologyContainer::ControlPointLocation(rankTriangle,trbia[j])));
					}
					rankTriangle++;
				}
			}
			// copy the weights 
			const BezierTetrahedronSetTopologyContainer::SeqWeights &swFrom=from_btstc->getWeightArray();

			BezierTriangleSetTopologyContainer::SeqWeights &wa=*(to_btstc->d_weightArray.beginEdit());
			wa.resize(swFrom.size());
			std::copy(swFrom.begin(),swFrom.end(),wa.begin());
			to_btstc->d_weightArray.endEdit();
			
			to_btstc->d_numberOfTriangularPoints.setValue(vertexSet.size());
			to_btstc->checkTopology();
			//to_tstm->propagateTopologicalChanges();
			to_tstm->notifyEndingEvent();
			//to_tstm->propagateTopologicalChanges();
			Loc2GlobDataVec.endEdit();
		}
	}
}


unsigned int BezierTetra2BezierTriangleTopologicalMapping::getFromIndex(unsigned int ind)
{

    if(fromModel->getTetrahedraAroundTriangle(ind).size()==1)
    {
        return fromModel->getTetrahedraAroundTriangle(ind)[0];
    }
    else
    {
        return 0;
    }
}

void BezierTetra2BezierTriangleTopologicalMapping::updateTopologicalMappingTopDown()
{

    // INITIALISATION of TRIANGULAR mesh from TETRAHEDRAL mesh :
//	cerr << "updateTopologicalMappingTopDown called" << endl;

    if (fromModel)
    {

        TriangleSetTopologyModifier *to_tstm;
        toModel->getContext()->get(to_tstm);

        if (toModel)
        {

            std::list<const TopologyChange *>::const_iterator itBegin=fromModel->beginChange();
            std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

            //sofa::helper::vector <unsigned int>& Loc2GlobVec = *(Loc2GlobDataVec.beginEdit());

            while( itBegin != itEnd )
            {
                TopologyChangeType changeType = (*itBegin)->getChangeType();

                switch( changeType )
                {

                case core::topology::ENDING_EVENT:
                {
                    //sout << "INFO_print : Tetra2TriangleTopologicalMapping - ENDING_EVENT" << sendl;
                    to_tstm->propagateTopologicalChanges();
                    to_tstm->notifyEndingEvent();
                    to_tstm->propagateTopologicalChanges();
                    break;
                }

                case core::topology::TRIANGLESREMOVED:
                {

                    break;
                }

                case core::topology::TRIANGLESADDED:
                {
                   
                    break;
                }

                case core::topology::TETRAHEDRAADDED:
                {
                    
                    break;
                }

                case core::topology::TETRAHEDRAREMOVED:
                {
                    

                    break;

                }

                case core::topology::EDGESADDED:
                {
                    
                    break;
                }

                case core::topology::POINTSADDED:
                {

                  
                    break;
                }

                case core::topology::POINTSREMOVED:
                {
                   

                    break;
                }

                case core::topology::POINTSRENUMBERING:
                {
                    

                    break;
                }
                default:
                    // Ignore events that are not Triangle  related.
                    break;
                };

                ++itBegin;
            }
            to_tstm->propagateTopologicalChanges();
            //Loc2GlobDataVec.endEdit();
        }
    }

    return;
}


} // namespace topology

} // namespace component

} // namespace sofa
