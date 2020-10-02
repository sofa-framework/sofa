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
#ifndef SOFA_COMPONENT_MASS_MESHMATRIXMASS_INL
#define SOFA_COMPONENT_MASS_MESHMATRIXMASS_INL

#include <SofaMiscForceField/MeshMatrixMass.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseMechanics/AddMToMatrixFunctor.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <SofaGeneralTopology/BezierTetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>

#ifdef SOFA_SUPPORT_MOVING_FRAMES
#include <sofa/core/behavior/InertiaForce.h>
#endif

namespace sofa
{

namespace component
{

namespace mass
{

template <class DataTypes, class MassType>
MeshMatrixMass<DataTypes, MassType>::MeshMatrixMass()
    : vertexMassInfo( initData(&vertexMassInfo, "vertexMass", "values of the particles masses on vertices") )
    , edgeMassInfo( initData(&edgeMassInfo, "edgeMass", "values of the particles masses on edges") )
    , tetrahedronMassInfo( initData(&tetrahedronMassInfo, "tetrahedronMass", "values of the particles masses for all control points inside a Bezier tetrahedron") )
    , m_massDensity( initData(&m_massDensity, (Real)1.0,"massDensity", "mass density that allows to compute the  particles masses from a mesh topology and geometry.\nOnly used if > 0") )
    , showCenterOfGravity( initData(&showCenterOfGravity, false, "showGravityCenter", "display the center of gravity of the system" ) )
    , showAxisSize( initData(&showAxisSize, (Real)1.0, "showAxisSizeFactor", "factor length of the axis displayed (only used for rigids)" ) )
    , lumping( initData(&lumping, true, "lumping","boolean if you need to use a lumped mass matrix") )
    , printMass( initData(&printMass, false, "printMass","boolean if you want to get the totalMass") )
    , f_graph( initData(&f_graph,"graph","Graph of the controlled potential") )
    , numericalIntegrationOrder( initData(&numericalIntegrationOrder,(size_t)2,"integrationOrder","The order of integration for numerical integration"))
    , numericalIntegrationMethod( initData(&numericalIntegrationMethod,(size_t)0,"numericalIntegrationMethod","The type of numerical integration method chosen"))
    , d_integrationMethod( initData(&d_integrationMethod,std::string("analytical"),"integrationMethod","\"exact\" if closed form expression for high order elements, \"analytical\" if closed form expression for affine element, \"numerical\" if numerical integration is chosen"))
    , topologyType(TOPOLOGY_UNKNOWN)
    , vertexMassHandler(NULL)
    , edgeMassHandler(NULL)
    , tetrahedronMassHandler(NULL)
{
    f_graph.setWidget("graph");
}

template <class DataTypes, class MassType>
MeshMatrixMass<DataTypes, MassType>::~MeshMatrixMass()
{
    if (vertexMassHandler) delete vertexMassHandler;
    if (edgeMassHandler) delete edgeMassHandler;
	if (tetrahedronMassHandler) delete tetrahedronMassHandler;
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyCreateFunction(unsigned int, MassType & VertexMass,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    VertexMass = 0;
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyCreateFunction(unsigned int, MassType & EdgeMass,
        const core::topology::BaseMeshTopology::Edge&,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    EdgeMass = 0;
}
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::TetrahedronMassHandler::applyCreateFunction(unsigned int tetra, MassVector & TetrahedronMass,
        const core::topology::BaseMeshTopology::Tetrahedron&,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
	MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

	if (MMM && (MMM->bezierTetraGeo) && (MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_BEZIERTETRAHEDRONSET))
	{	
		Real densityM = MMM->getMassDensity();
		topology::BezierDegreeType degree=MMM->bezierTetraGeo->getTopologyContainer()->getDegree();
		size_t nbControlPoints=(degree+1)*(degree+2)*(degree+3)/6;
		size_t nbMassEntries=nbControlPoints*(nbControlPoints+1)/2;

		if (TetrahedronMass.size()!=nbMassEntries) {
			TetrahedronMass.resize(nbMassEntries);
		}
		// set array to zero
		std::fill(TetrahedronMass.begin(),TetrahedronMass.end(),(MassType)0);
		sofa::helper::vector<MassType> lumpedVertexMass;
		lumpedVertexMass.resize(nbControlPoints);
		size_t i,j,k,rank;
		typedef typename topology::BezierTetrahedronSetGeometryAlgorithms<DataTypes>::VecPointID VecPointID;
		/// get the global index of each control point in the tetrahedron
		const VecPointID &indexArray=MMM->bezierTetraGeo->getTopologyContainer()->getGlobalIndexArrayOfBezierPoints(tetra);
		std::fill(lumpedVertexMass.begin(),lumpedVertexMass.end(),(MassType)0);
		if (MMM->integrationMethod==MeshMatrixMass<DataTypes, MassType>::NUMERICAL_INTEGRATION) {
			sofa::helper::vector<Real> shapeFunctionValue;
			shapeFunctionValue.resize(nbControlPoints);
			// set array to zero

			/// get value of integration points
			topology::NumericalIntegrationDescriptor<Real,4> &nid=MMM->bezierTetraGeo->getTetrahedronNumericalIntegrationDescriptor();
            typename topology::NumericalIntegrationDescriptor<Real,4>::QuadraturePointArray qpa=nid.getQuadratureMethod((typename topology::NumericalIntegrationDescriptor<Real,4>::QuadratureMethod)MMM->numericalIntegrationMethod.getValue(),
				MMM->numericalIntegrationOrder.getValue());

			sofa::defaulttype::Vec<4,Real> bc;
			sofa::helper::vector<topology::TetrahedronBezierIndex> tbi=MMM->bezierTetraGeo->getTopologyContainer()->getTetrahedronBezierIndexArray();
			typename DataTypes::Real jac,weight;
			MassType tmpMass;

			// loop through the integration points
			for (i=0;i<qpa.size();++i) {
                typename topology::NumericalIntegrationDescriptor<Real,4>::QuadraturePoint qp=qpa[i];
				// the barycentric coordinate
				bc=qp.first;
				// the weight of the integration point
				weight=qp.second;
				// the Jacobian Derterminant of the integration point 
				jac=MMM->bezierTetraGeo->computeJacobian(tetra,bc)*densityM;
				/// prestore the shape function value for that integration point.
				for (j=0;j<nbControlPoints;j++) {
					shapeFunctionValue[j]=MMM->bezierTetraGeo->computeBernsteinPolynomial(tbi[j],bc);
				}
				// now loop through each pair of control point to compute the mass
				rank=0;
				for (j=0;j<nbControlPoints;j++) {
					// use the fact that the shapefunctions sum to 1 to get the lumped value
					lumpedVertexMass[j]+=shapeFunctionValue[j]*fabs(jac)*weight;

					for (k=j;k<nbControlPoints;k++,rank++) {
						/// compute the mass as the integral of product of the 2 shape functions multiplied by the Jacobian
						tmpMass=shapeFunctionValue[k]*shapeFunctionValue[j]*fabs(jac)*weight;
						TetrahedronMass[rank]+=tmpMass;
					}
				}
			}
	//		std::cerr<<"Mass Matrix= "<<TetrahedronMass<<std::endl;
	//		std::cerr<<"Lumped Mass Matrix= "<<lumpedVertexMass<<std::endl;
			// now updates the the mass matrix on each vertex.
            helper::vector<MassType>& my_vertexMassInfo = *MMM->vertexMassInfo.beginEdit();
			for (j=0;j<nbControlPoints;j++) {
				my_vertexMassInfo[indexArray[j]]+=lumpedVertexMass[j];
			}
		} else if ((MMM->integrationMethod==MeshMatrixMass<DataTypes, MassType>::AFFINE_ELEMENT_INTEGRATION) || 
            (MMM->bezierTetraGeo->isBezierTetrahedronAffine(tetra,(MMM->bezierTetraGeo->getDOF()->read(core::ConstVecCoordId::restPosition())->getValue()) )))
		{
			/// affine mass simple computation
			
			Real totalMass= densityM*MMM->tetraGeo->computeRestTetrahedronVolume(tetra);
			Real mass=totalMass/(topology::binomial<typename DataTypes::Real>(degree,degree)*topology::binomial<typename DataTypes::Real>(2*degree,3));
			sofa::helper::vector<topology::TetrahedronBezierIndex> tbiArray;
			topology::TetrahedronBezierIndex tbi1,tbi2;
			tbiArray=MMM->bezierTetraGeo->getTopologyContainer()->getTetrahedronBezierIndexArray();
			rank=0;
			for (j=0;j<nbControlPoints;j++) {
				tbi1=tbiArray[j];
				for (k=j;k<nbControlPoints;k++,rank++) {
					tbi2=tbiArray[k];
					TetrahedronMass[rank]+=mass*topology::binomialVector<4,typename DataTypes::Real>(tbi1,tbi2);
	//				std::cerr<<" tbi = "<<tbi1<<" "<<tbi2<<" ="<<TetrahedronMass[rank]<<std::endl;
				}
			}
			// mass for mass lumping
			mass=totalMass/nbControlPoints;
			// now updates the the mass matrix on each vertex.
            helper::vector<MassType>& my_vertexMassInfo = *MMM->vertexMassInfo.beginEdit();
			for (j=0;j<nbControlPoints;j++) {
				my_vertexMassInfo[indexArray[j]]+=mass;
			}
		} else {
			/// exact computation
			sofa::helper::vector<topology::TetrahedronBezierIndex> tbiArray,tbiDerivArray,multinomialArray;
			sofa::helper::vector<unsigned char> multinomialScalarArray;
			/// use the rest configuration
//			const typename DataTypes::VecCoord &p=(MMM->bezierTetraGeo->getDOF()->read(core::ConstVecCoordId::restPosition())->getValue());

			Real factor;
			MassType tmpMass;

			tbiArray=MMM->bezierTetraGeo->getTopologyContainer()->getTetrahedronBezierIndexArray();
			tbiDerivArray=MMM->bezierTetraGeo->getTopologyContainer()->getTetrahedronBezierIndexArrayOfGivenDegree(degree-1);
			sofa::helper::vector<topology::BezierTetrahedronSetTopologyContainer::LocalTetrahedronIndex> correspondanceArray=
				MMM->bezierTetraGeo->getTopologyContainer()->getMapOfTetrahedronBezierIndexArrayFromInferiorDegree();
			typename DataTypes::Coord dp1,dp2,dp3,dpos,tmp;
			multinomialArray.resize(5);
			multinomialScalarArray.resize(5);
			multinomialScalarArray[0]=degree-1;
			multinomialScalarArray[1]=degree-1;
			multinomialScalarArray[2]=degree-1;
			multinomialScalarArray[3]=degree;
			multinomialScalarArray[4]=degree;
			size_t l,m;
			factor=6*topology::multinomial<Real>(5*degree-3,multinomialScalarArray)*topology::binomial<Real>(5*degree-3,3)/(degree*degree*degree*densityM);
			for (i=0;i<tbiDerivArray.size();++i) 
			{
				dp1=MMM->bezierTetraGeo->getPointRestPosition(indexArray[correspondanceArray[i][0]])-MMM->bezierTetraGeo->getPointRestPosition(indexArray[correspondanceArray[i][3]]);
				multinomialArray[0]=tbiDerivArray[i];
				for (j=0;j<tbiDerivArray.size();++j) 
				{
					dp2=MMM->bezierTetraGeo->getPointRestPosition(indexArray[correspondanceArray[j][1]])-
						MMM->bezierTetraGeo->getPointRestPosition(indexArray[correspondanceArray[j][3]]);
                    using topology::cross;
					tmp=cross<Real>(dp1,dp2);
					multinomialArray[1]=tbiDerivArray[j];
					rank=0;
					for (l=0;l<nbControlPoints;l++) {
						multinomialArray[3]=tbiArray[l];
						for (m=l;m<nbControlPoints;m++,rank++) {
							multinomialArray[4]=tbiArray[m];
							// set dp3 to 0 in a generic way
							std::fill(dp3.begin(),dp3.end(),(Real)0);
							for (k=0;k<tbiDerivArray.size();++k) 
							{
								multinomialArray[2]=tbiDerivArray[k];
								dpos=MMM->bezierTetraGeo->getPointRestPosition(indexArray[correspondanceArray[k][2]])-
									MMM->bezierTetraGeo->getPointRestPosition(indexArray[correspondanceArray[k][3]]);
								dpos*=topology::multinomialVector<4,Real>(multinomialArray);
								dp3+=dpos;
							}
							dp3/=factor;
							tmpMass=fabs(dp3*tmp);
							TetrahedronMass[rank]+=tmpMass;
							lumpedVertexMass[l]+=tmpMass;
							if (m>l)
								lumpedVertexMass[m]+=tmpMass;
						}

					}
				}

			}

			// now updates the the mass matrix on each vertex.
            helper::vector<MassType>& my_vertexMassInfo = *MMM->vertexMassInfo.beginEdit();
			for (j=0;j<nbControlPoints;j++) {
				my_vertexMassInfo[indexArray[j]]+=lumpedVertexMass[j];
			}
		}
	}
}

// -------------------------------------------------------
// ------- Triangle Creation/Destruction functions -------
// -------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyTriangleCreation(const sofa::helper::vector< unsigned int >& triangleAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Triangle >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->vertexMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleAdded.size(); ++i)
        {
            // Get the triangle to be added
            const core::topology::BaseMeshTopology::Triangle &t = MMM->_topology->getTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                mass=(densityM * MMM->triangleGeo->computeRestTriangleArea(triangleAdded[i]))/(typename DataTypes::Real)6.0;
            }

            // Adding mass
            for (unsigned int j=0; j<3; ++j)
                VertexMasses[ t[j] ] += mass;
        }
    }
}

/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyTriangleCreation(const sofa::helper::vector< unsigned int >& triangleAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Triangle >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->edgeMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleAdded.size(); ++i)
        {
            // Get the edgesInTriangle to be added
            const core::topology::BaseMeshTopology::EdgesInTriangle &te = MMM->_topology->getEdgesInTriangle(triangleAdded[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                mass=(densityM * MMM->triangleGeo->computeRestTriangleArea(triangleAdded[i]))/(typename DataTypes::Real)12.0;
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<3; ++j)
                EdgeMasses[ te[j] ] += mass;
        }
    }
}


/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyTriangleDestruction(const sofa::helper::vector< unsigned int >& triangleRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->vertexMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const core::topology::BaseMeshTopology::Triangle &t = MMM->_topology->getTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                mass=(densityM * MMM->triangleGeo->computeRestTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real)6.0;
            }

            // Removing mass
            for (unsigned int j=0; j<3; ++j)
                VertexMasses[ t[j] ] -= mass;
        }
    }
}


/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyTriangleDestruction(const sofa::helper::vector< unsigned int >& triangleRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TRIANGLESET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->edgeMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<triangleRemoved.size(); ++i)
        {
            // Get the triangle to be removed
            const core::topology::BaseMeshTopology::EdgesInTriangle &te = MMM->_topology->getEdgesInTriangle(triangleRemoved[i]);

            // Compute rest mass of conserne triangle = density * triangle surface.
            if(MMM->triangleGeo)
            {
                mass=(densityM * MMM->triangleGeo->computeRestTriangleArea(triangleRemoved[i]))/(typename DataTypes::Real)12.0;
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<3; ++j)
                EdgeMasses[ te[j] ] -= mass;
        }
    }
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::TrianglesAdded* e)
{
    const sofa::helper::vector<unsigned int> &triangleAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTriangleCreation(triangleAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::TrianglesRemoved* e)
{
    const sofa::helper::vector<unsigned int> &triangleRemoved = e->getArray();

    applyTriangleDestruction(triangleRemoved);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::TrianglesAdded* e)
{
    const sofa::helper::vector<unsigned int> &triangleAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Triangle> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTriangleCreation(triangleAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::TrianglesRemoved* e)
{
    const sofa::helper::vector<unsigned int> &triangleRemoved = e->getArray();

    applyTriangleDestruction(triangleRemoved);
}

// }

// ---------------------------------------------------
// ------- Quad Creation/Destruction functions -------
// ---------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyQuadCreation(const sofa::helper::vector< unsigned int >& quadAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Quad >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->vertexMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadAdded.size(); ++i)
        {
            // Get the quad to be added
            const core::topology::BaseMeshTopology::Quad &q = MMM->_topology->getQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                mass=(densityM * MMM->quadGeo->computeRestQuadArea(quadAdded[i]))/(typename DataTypes::Real)8.0;
            }

            // Adding mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ q[j] ] += mass;
        }
    }
}


/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyQuadCreation(const sofa::helper::vector< unsigned int >& quadAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Quad >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->edgeMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadAdded.size(); ++i)
        {
            // Get the EdgesInQuad to be added
            const core::topology::BaseMeshTopology::EdgesInQuad &qe = MMM->_topology->getEdgesInQuad(quadAdded[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                mass=(densityM * MMM->quadGeo->computeRestQuadArea(quadAdded[i]))/(typename DataTypes::Real)16.0;
            }

            // Adding mass edges of concerne quad
            for (unsigned int j=0; j<4; ++j)
                EdgeMasses[ qe[j] ] += mass;
        }
    }
}


/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyQuadDestruction(const sofa::helper::vector< unsigned int >& quadRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->vertexMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadRemoved.size(); ++i)
        {
            // Get the quad to be removed
            const core::topology::BaseMeshTopology::Quad &q = MMM->_topology->getQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                mass=(densityM * MMM->quadGeo->computeRestQuadArea(quadRemoved[i]))/(typename DataTypes::Real)8.0;
            }

            // Removing mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ q[j] ] -= mass;
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyQuadDestruction(const sofa::helper::vector< unsigned int >& quadRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_QUADSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->edgeMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<quadRemoved.size(); ++i)
        {
            // Get the EdgesInQuad to be removed
            const core::topology::BaseMeshTopology::EdgesInQuad &qe = MMM->_topology->getEdgesInQuad(quadRemoved[i]);

            // Compute rest mass of conserne quad = density * quad surface.
            if(MMM->quadGeo)
            {
                mass=(densityM * MMM->quadGeo->computeRestQuadArea(quadRemoved[i]))/(typename DataTypes::Real)16.0;
            }

            // Removing mass edges of concerne quad
            for (unsigned int j=0; j<4; ++j)
                EdgeMasses[ qe[j] ] -= mass/2;
        }
    }
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::QuadsAdded* e)
{
    const sofa::helper::vector<unsigned int> &quadAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyQuadCreation(quadAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::QuadsRemoved* e)
{
    const sofa::helper::vector<unsigned int> &quadRemoved = e->getArray();

    applyQuadDestruction(quadRemoved);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::QuadsAdded* e)
{
    const sofa::helper::vector<unsigned int> &quadAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Quad> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyQuadCreation(quadAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::QuadsRemoved* e)
{
    const sofa::helper::vector<unsigned int> &quadRemoved = e->getArray();

    applyQuadDestruction(quadRemoved);
}

// }



// ----------------------------------------------------------
// ------- Tetrahedron Creation/Destruction functions -------
// ----------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& tetrahedronAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Tetrahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->vertexMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronAdded.size(); ++i)
        {
            // Get the tetrahedron to be added
            const core::topology::BaseMeshTopology::Tetrahedron &t = MMM->_topology->getTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                mass=(densityM * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real)10.0;
            }

            // Adding mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ t[j] ] += mass;
        }
    }
}


/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& tetrahedronAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Tetrahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->edgeMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronAdded.size(); ++i)
        {
            // Get the edgesInTetrahedron to be added
            const core::topology::BaseMeshTopology::EdgesInTetrahedron &te = MMM->_topology->getEdgesInTetrahedron(tetrahedronAdded[i]);

            // Compute rest mass of conserne triangle = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                mass=(densityM * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronAdded[i]))/(typename DataTypes::Real)20.0;
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<6; ++j)
                EdgeMasses[ te[j] ] += mass;
        }
    }
}


/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyTetrahedronDestruction(const sofa::helper::vector< unsigned int >& tetrahedronRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->vertexMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronRemoved.size(); ++i)
        {
            // Get the tetrahedron to be removed
            const core::topology::BaseMeshTopology::Tetrahedron &t = MMM->_topology->getTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne tetrahedron = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                mass=(densityM * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real)10.0;
            }

            // Removing mass
            for (unsigned int j=0; j<4; ++j)
                VertexMasses[ t[j] ] -= mass;
        }
    }
}

/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyTetrahedronDestruction(const sofa::helper::vector< unsigned int >& tetrahedronRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_TETRAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->edgeMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<tetrahedronRemoved.size(); ++i)
        {
            // Get the edgesInTetrahedron to be removed
            const core::topology::BaseMeshTopology::EdgesInTetrahedron &te = MMM->_topology->getEdgesInTetrahedron(tetrahedronRemoved[i]);

            // Compute rest mass of conserne triangle = density * tetrahedron volume.
            if(MMM->tetraGeo)
            {
                mass=(densityM * MMM->tetraGeo->computeRestTetrahedronVolume(tetrahedronRemoved[i]))/(typename DataTypes::Real)20.0;
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<6; ++j)
                EdgeMasses[ te[j] ] -= mass; //?
        }
    }
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::TetrahedraAdded* e)
{
    const sofa::helper::vector<unsigned int> &tetraAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTetrahedronCreation(tetraAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::TetrahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> &tetraRemoved = e->getArray();

    applyTetrahedronDestruction(tetraRemoved);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::TetrahedraAdded* e)
{
    const sofa::helper::vector<unsigned int> &tetraAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Tetrahedron> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyTetrahedronCreation(tetraAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::TetrahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> &tetraRemoved = e->getArray();

    applyTetrahedronDestruction(tetraRemoved);
}

// }


// ---------------------------------------------------------
// ------- Hexahedron Creation/Destruction functions -------
// ---------------------------------------------------------
//{

/// Creation fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyHexahedronCreation(const sofa::helper::vector< unsigned int >& hexahedronAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Hexahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->vertexMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronAdded.size(); ++i)
        {
            // Get the hexahedron to be added
            const core::topology::BaseMeshTopology::Hexahedron &h = MMM->_topology->getHexahedron(hexahedronAdded[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                mass=(densityM * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real)20.0;
            }

            // Adding mass
            for (unsigned int j=0; j<8; ++j)
                VertexMasses[ h[j] ] += mass;
        }
    }
}


/// Creation fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyHexahedronCreation(const sofa::helper::vector< unsigned int >& hexahedronAdded,
        const sofa::helper::vector< core::topology::BaseMeshTopology::Hexahedron >& /*elems*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;

    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->edgeMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronAdded.size(); ++i)
        {
            // Get the EdgesInHexahedron to be added
            const core::topology::BaseMeshTopology::EdgesInHexahedron &he = MMM->_topology->getEdgesInHexahedron(hexahedronAdded[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                mass=(densityM * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronAdded[i]))/(typename DataTypes::Real)40.0;
            }

            // Adding mass edges of concerne triangle
            for (unsigned int j=0; j<12; ++j)
                EdgeMasses[ he[j] ] += mass;
        }
    }
}


/// Destruction fonction for mass stored on vertices
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::applyHexahedronDestruction(const sofa::helper::vector< unsigned int >& hexahedronRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > VertexMasses ( MMM->vertexMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronRemoved.size(); ++i)
        {
            // Get the hexahedron to be removed
            const core::topology::BaseMeshTopology::Hexahedron &h = MMM->_topology->getHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                mass=(densityM * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real)20.0;
            }

            // Removing mass
            for (unsigned int j=0; j<8; ++j)
                VertexMasses[ h[j] ] -= mass;
        }
    }
}


/// Destruction fonction for mass stored on edges
template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::applyHexahedronDestruction(const sofa::helper::vector< unsigned int >& hexahedronRemoved)
{
    MeshMatrixMass<DataTypes, MassType> *MMM = this->m;
    if (MMM && MMM->getMassTopologyType()==MeshMatrixMass<DataTypes, MassType>::TOPOLOGY_HEXAHEDRONSET)
    {
        helper::WriteAccessor< Data< helper::vector<MassType> > > EdgeMasses ( MMM->edgeMassInfo );
        // Initialisation
        typename DataTypes::Real densityM = MMM->getMassDensity();
        typename DataTypes::Real mass = (typename DataTypes::Real) 0;

        for (unsigned int i = 0; i<hexahedronRemoved.size(); ++i)
        {
            // Get the EdgesInHexahedron to be removed
            const core::topology::BaseMeshTopology::EdgesInHexahedron &he = MMM->_topology->getEdgesInHexahedron(hexahedronRemoved[i]);

            // Compute rest mass of conserne hexahedron = density * hexahedron volume.
            if(MMM->hexaGeo)
            {
                mass=(densityM * MMM->hexaGeo->computeRestHexahedronVolume(hexahedronRemoved[i]))/(typename DataTypes::Real)40.0;
            }

            // Removing mass edges of concerne triangle
            for (unsigned int j=0; j<12; ++j)
                EdgeMasses[ he[j] ] -= mass;
        }
    }
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::HexahedraAdded* e)
{
    const sofa::helper::vector<unsigned int> &hexaAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyHexahedronCreation(hexaAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::VertexMassHandler::ApplyTopologyChange(const core::topology::HexahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> &hexaRemoved = e->getArray();

    applyHexahedronDestruction(hexaRemoved);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::HexahedraAdded* e)
{
    const sofa::helper::vector<unsigned int> &hexaAdded = e->getIndexArray();
    const sofa::helper::vector<core::topology::BaseMeshTopology::Hexahedron> &elems = e->getElementArray();
    const sofa::helper::vector<sofa::helper::vector<unsigned int> > & ancestors = e->ancestorsList;
    const sofa::helper::vector<sofa::helper::vector<double> > & coefs = e->coefs;

    applyHexahedronCreation(hexaAdded, elems, ancestors, coefs);
}

template< class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::EdgeMassHandler::ApplyTopologyChange(const core::topology::HexahedraRemoved* e)
{
    const sofa::helper::vector<unsigned int> &hexaRemoved = e->getArray();

    applyHexahedronDestruction(hexaRemoved);
}

// }



template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::init()
{
    /*  using sofa::component::topology::RegularGridTopology;
    RegularGridTopology* reg = dynamic_cast<RegularGridTopology*>( this->getContext()->getMeshTopology() );
    if( reg != NULL )
    {
    Real weight = reg->getDx().norm() * reg->getDy().norm() * reg->getDz().norm() * m_massDensity.getValue()/8;
    VecMass& m = *f_mass.beginEdit();
    for( int i=0; i<reg->getNx()-1; i++ )
    {
    for( int j=0; j<reg->getNy()-1; j++ )
    {
        for( int k=0; k<reg->getNz()-1; k++ )
        {
    m[reg->point(i,j,k)] += weight;
    m[reg->point(i,j,k+1)] += weight;
    m[reg->point(i,j+1,k)] += weight;
    m[reg->point(i,j+1,k+1)] += weight;
    m[reg->point(i+1,j,k)] += weight;
    m[reg->point(i+1,j,k+1)] += weight;
    m[reg->point(i+1,j+1,k)] += weight;
    m[reg->point(i+1,j+1,k+1)] += weight;
        }
    }
    }
    f_mass.endEdit();
    }*/

    this->Inherited::init();
    massLumpingCoeff = 0.0;

	if (d_integrationMethod.getValue() == "analytical")
		integrationMethod= AFFINE_ELEMENT_INTEGRATION;
	else if (d_integrationMethod.getValue() == "numerical") 
		integrationMethod= NUMERICAL_INTEGRATION;
	else if (d_integrationMethod.getValue() == "exact") 
		integrationMethod= EXACT_INTEGRATION;
	else
	{
		serr << "cannot recognize method "<< d_integrationMethod.getValue() << ". Must be either  \"exact\", \"analytical\" or \"numerical\"" << sendl;
	}

    _topology = this->getContext()->getMeshTopology();
    savedMass = m_massDensity.getValue();

    //    sofa::core::objectmodel::Tag mechanicalTag(m_tagMeshMechanics.getValue());
    //    this->getContext()->get(triangleGeo, mechanicalTag,sofa::core::objectmodel::BaseContext::SearchUp);

    this->getContext()->get(edgeGeo);
    this->getContext()->get(triangleGeo);
    this->getContext()->get(quadGeo);
    this->getContext()->get(tetraGeo);
    this->getContext()->get(hexaGeo);
	this->getContext()->get(bezierTetraGeo);

    // add the functions to handle topology changes for Vertex informations
    vertexMassHandler = new VertexMassHandler(this, &vertexMassInfo);
    vertexMassInfo.createTopologicalEngine(_topology, vertexMassHandler);
    vertexMassInfo.linkToEdgeDataArray();
    vertexMassInfo.linkToTriangleDataArray();
    vertexMassInfo.linkToQuadDataArray();
    vertexMassInfo.linkToTetrahedronDataArray();
    vertexMassInfo.linkToHexahedronDataArray();
    vertexMassInfo.registerTopologicalData();

    // add the functions to handle topology changes for Edge informations
    edgeMassHandler = new EdgeMassHandler(this, &edgeMassInfo);
    edgeMassInfo.createTopologicalEngine(_topology, edgeMassHandler);
    edgeMassInfo.linkToTriangleDataArray();
    edgeMassInfo.linkToQuadDataArray();
    edgeMassInfo.linkToTetrahedronDataArray();
    edgeMassInfo.linkToHexahedronDataArray();
    edgeMassInfo.registerTopologicalData();

	if (bezierTetraGeo) {
		// for Bezier Tetrahedra add the functions to handle topology changes for Tetrahedron informations
		tetrahedronMassHandler = new TetrahedronMassHandler(this, &tetrahedronMassInfo);
		tetrahedronMassInfo.createTopologicalEngine(_topology, tetrahedronMassHandler);
		tetrahedronMassInfo.linkToTetrahedronDataArray();

	}

    if ((vertexMassInfo.getValue().size()==0 || edgeMassInfo.getValue().size()==0) && (_topology!=0))
        reinit();

    //Reset the graph
    f_graph.beginEdit()->clear();
    f_graph.endEdit();

    this->copyVertexMass();
}

template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::reinit()
{
    if (_topology && ((m_massDensity.getValue() > 0 && (vertexMassInfo.getValue().size() == 0 || edgeMassInfo.getValue().size() == 0)) || (m_massDensity.getValue()!= savedMass) ))
    {
        // resize array
        clear();

        /// prepare to store info in the vertex array
        helper::vector<MassType>& my_vertexMassInfo = *vertexMassInfo.beginEdit();
        helper::vector<MassType>& my_edgeMassInfo = *edgeMassInfo.beginEdit();

        unsigned int ndof = this->mstate->getSize();
        unsigned int nbEdges=_topology->getNbEdges();
        const helper::vector<core::topology::BaseMeshTopology::Edge>& edges = _topology->getEdges();

        my_vertexMassInfo.resize(ndof);
        my_edgeMassInfo.resize(nbEdges);

        const helper::vector< unsigned int > emptyAncestor;
        const helper::vector< double > emptyCoefficient;
        const helper::vector< helper::vector< unsigned int > > emptyAncestors;
        const helper::vector< helper::vector< double > > emptyCoefficients;

        // set vertex tensor to 0
        for (unsigned int i = 0; i<ndof; ++i)
            vertexMassHandler->applyCreateFunction(i, my_vertexMassInfo[i], emptyAncestor, emptyCoefficient);

        // set edge tensor to 0
        for (unsigned int i = 0; i<nbEdges; ++i)
            edgeMassHandler->applyCreateFunction(i, my_edgeMassInfo[i], edges[i], emptyAncestor, emptyCoefficient);

        // Create mass matrix depending on current Topology:
        if (_topology->getNbHexahedra()>0 && hexaGeo)  // Hexahedron topology
        {
            // create vector tensor by calling the hexahedron creation function on the entire mesh
            sofa::helper::vector<unsigned int> hexahedraAdded;
            setMassTopologyType(TOPOLOGY_HEXAHEDRONSET);
            int n = _topology->getNbHexahedra();
            for (int i = 0; i<n; ++i)
                hexahedraAdded.push_back(i);

            vertexMassHandler->applyHexahedronCreation(hexahedraAdded, _topology->getHexahedra(), emptyAncestors, emptyCoefficients);
            edgeMassHandler->applyHexahedronCreation(hexahedraAdded, _topology->getHexahedra(), emptyAncestors, emptyCoefficients);
            massLumpingCoeff = 2.5;
        }
        

		else if (_topology->getNbTetrahedra()>0 && bezierTetraGeo)  // Bezier Tetrahedron topology
        {
            helper::vector<MassVector>& my_tetrahedronMassInfo = *tetrahedronMassInfo.beginEdit();


			size_t  nbTetrahedra=_topology->getNbTetrahedra();
            const helper::vector<core::topology::BaseMeshTopology::Tetra>& tetrahedra = _topology->getTetrahedra();

			my_tetrahedronMassInfo.resize(nbTetrahedra);
			 setMassTopologyType(TOPOLOGY_BEZIERTETRAHEDRONSET);
			// set vertex tensor to 0
			for (unsigned int i = 0; i<nbTetrahedra; ++i)
				tetrahedronMassHandler->applyCreateFunction(i, my_tetrahedronMassInfo[i], tetrahedra[i],emptyAncestor, emptyCoefficient);

            // create vector tensor by calling the tetrahedron creation function on the entire mesh
            sofa::helper::vector<unsigned int> tetrahedraAdded;
           

			size_t n = _topology->getNbTetrahedra();
			for (size_t i = 0; i<n; ++i)
				tetrahedraAdded.push_back(i);

//			tetrahedronMassHandler->applyTetrahedronCreation(tetrahedraAdded, _topology->getTetrahedra(), _topology->getTetrahedron(i),emptyAncestors, emptyCoefficients);
			massLumpingCoeff = 1.0;

			tetrahedronMassInfo.registerTopologicalData();
			tetrahedronMassInfo.endEdit();
        }
		else if (_topology->getNbTetrahedra()>0 && tetraGeo)  // Tetrahedron topology
        {
            // create vector tensor by calling the tetrahedron creation function on the entire mesh
            sofa::helper::vector<unsigned int> tetrahedraAdded;
            setMassTopologyType(TOPOLOGY_TETRAHEDRONSET);

            int n = _topology->getNbTetrahedra();
            for (int i = 0; i<n; ++i)
                tetrahedraAdded.push_back(i);

            vertexMassHandler->applyTetrahedronCreation(tetrahedraAdded, _topology->getTetrahedra(), emptyAncestors, emptyCoefficients);
            edgeMassHandler->applyTetrahedronCreation(tetrahedraAdded, _topology->getTetrahedra(), emptyAncestors, emptyCoefficients);
            massLumpingCoeff = 2.5;
        }
        else if (_topology->getNbQuads()>0 && quadGeo)  // Quad topology
        {
            // create vector tensor by calling the quad creation function on the entire mesh
            sofa::helper::vector<unsigned int> quadsAdded;
            setMassTopologyType(TOPOLOGY_QUADSET);

            int n = _topology->getNbQuads();
            for (int i = 0; i<n; ++i)
                quadsAdded.push_back(i);

            vertexMassHandler->applyQuadCreation(quadsAdded, _topology->getQuads(), emptyAncestors, emptyCoefficients);
            edgeMassHandler->applyQuadCreation(quadsAdded, _topology->getQuads(), emptyAncestors, emptyCoefficients);
            massLumpingCoeff = 2.0;
        }
        else if (_topology->getNbTriangles()>0 && triangleGeo) // Triangle topology
        {
            // create vector tensor by calling the triangle creation function on the entire mesh
            sofa::helper::vector<unsigned int> trianglesAdded;
            setMassTopologyType(TOPOLOGY_TRIANGLESET);

            int n = _topology->getNbTriangles();
            for (int i = 0; i<n; ++i)
                trianglesAdded.push_back(i);

            vertexMassHandler->applyTriangleCreation(trianglesAdded, _topology->getTriangles(), emptyAncestors, emptyCoefficients);
            edgeMassHandler->applyTriangleCreation(trianglesAdded, _topology->getTriangles(), emptyAncestors, emptyCoefficients);
            massLumpingCoeff = 2.0;
        }

        vertexMassInfo.registerTopologicalData();
        edgeMassInfo.registerTopologicalData();

        vertexMassInfo.endEdit();
        edgeMassInfo.endEdit();
    }
}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::copyVertexMass() {}


template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::clear()
{
    MassVector& vertexMass = *vertexMassInfo.beginEdit();
    MassVector& edgeMass = *edgeMassInfo.beginEdit();
	MassVectorVector& tetrahedronMass = *tetrahedronMassInfo.beginEdit();
    vertexMass.clear();
    edgeMass.clear();
	tetrahedronMass.clear();
    vertexMassInfo.endEdit();
    edgeMassInfo.endEdit();
	tetrahedronMassInfo.endEdit();
}


// -- Mass interface
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addMDx(const core::MechanicalParams*, DataVecDeriv& vres, const DataVecDeriv& vdx, SReal factor)
{
    const MassVector &vertexMass= vertexMassInfo.getValue();
    const MassVector &edgeMass= edgeMassInfo.getValue();

    helper::WriteAccessor< DataVecDeriv > res = vres;
    helper::ReadAccessor< DataVecDeriv > dx = vdx;

    SReal massTotal = 0.0;

    //using a lumped matrix (default)-----
    if(this->lumping.getValue())
    {
        for (size_t i=0; i<dx.size(); i++)
        {
            res[i] += dx[i] * vertexMass[i] * massLumpingCoeff * (Real)factor;
            massTotal += vertexMass[i]*massLumpingCoeff * (Real)factor;
        }
        
    }


    //using a sparse matrix---------------
	else if (getMassTopologyType()!=TOPOLOGY_BEZIERTETRAHEDRONSET) 
	{
		size_t nbEdges=_topology->getNbEdges();
		size_t v0,v1;

		for (unsigned int i=0; i<dx.size(); i++)
		{
			res[i] += dx[i] * vertexMass[i] * (Real)factor;
			massTotal += vertexMass[i] * (Real)factor;
		}

		Real tempMass=0.0;

		for (unsigned int j=0; j<nbEdges; ++j)
		{
			tempMass = edgeMass[j] * (Real)factor;

			v0=_topology->getEdge(j)[0];
			v1=_topology->getEdge(j)[1];

			res[v0] += dx[v1] * tempMass;
			res[v1] += dx[v0] * tempMass;

			massTotal += 2*edgeMass[j] * (Real)factor;
		}
	} else if (bezierTetraGeo ){
		typedef typename topology::BezierTetrahedronSetGeometryAlgorithms<DataTypes>::VecPointID VecPointID;
			topology::BezierDegreeType degree=bezierTetraGeo->getTopologyContainer()->getDegree();
			size_t nbControlPoints=(degree+1)*(degree+2)*(degree+3)/6;
			size_t nbTetras=_topology->getNbTetrahedra();
#ifdef NDEBUG
			assert(tetrahedronMassInfo.size()==(nbControlPoints*(nbControlPoints+1)/2));
#endif
			// go through the mass stored in each tetrahedron element
			size_t rank=0;
			MassType tempMass;
			size_t v0,v1;
			// loop over each tetrahedron of size nbControlPoints*nbControlPoints
			for (size_t i=0; i<nbTetras; i++) {
				
				/// get the global index of each control point in the tetrahedron
				const VecPointID &indexArray=
					bezierTetraGeo->getTopologyContainer()->getGlobalIndexArrayOfBezierPoints(i) ;
				// get the mass matrix in the tetrahedron
//				const MassVector &mv=tetrahedronMassInfo.getValue()[i];
				const MassVector &mv=getBezierTetrahedronMassVector(i);
				nbControlPoints=indexArray.size();
				assert(mv.size()==nbControlPoints*(nbControlPoints+1)/2);
				// loop over each entry in the mass matrix of size nbControlPoints*(nbControlPoints+1)/2
				rank=0;
				for (size_t j=0; j<nbControlPoints; ++j) {
					v0 = indexArray[j];
					for (size_t k=j; k<nbControlPoints; ++k,++rank) {
						v1 = indexArray[k];
						tempMass =mv[rank] * (Real)factor;					
						if (k>j) {
							res[v0] += dx[v1] * tempMass;
							res[v1] += dx[v0] * tempMass;
							massTotal += 2*tempMass;
						} else {
							res[v0] += dx[v0] * tempMass;
							massTotal += tempMass;
						}
					}
				}
			}
		}
	if(printMass.getValue() && (this->getContext()->getTime()==0.0))
        sout<<"Total Mass = "<<massTotal<<sendl;

	if(printMass.getValue())
	{
		std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
		sofa::helper::vector<double>& graph_error = graph["Mass variations"];
		graph_error.push_back(massTotal+0.000001);

		f_graph.endEdit();
	}

        
    

}



template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::accFromF(const core::MechanicalParams*, DataVecDeriv& a, const DataVecDeriv& f)
{
    helper::WriteAccessor< DataVecDeriv > _a = a;
    const VecDeriv& _f = f.getValue();
    const MassVector &vertexMass= vertexMassInfo.getValue();

    if(this->lumping.getValue())
    {
        for (unsigned int i=0; i<vertexMass.size(); i++)
        {
            _a[i] = _f[i] / ( vertexMass[i] * massLumpingCoeff);
        }
    }
    else
    {
        (void)a;
        (void)f;
        serr << "WARNING: the methode 'accFromF' can't be used with MeshMatrixMass as this SPARSE mass matrix can't be inversed easily. \nPlease proceed to mass lumping." << sendl;
        return;
    }
}




#ifdef SOFA_SUPPORT_MOVING_FRAMES
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addForce(const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& vx, const DataVecDeriv& vv)
{
    helper::WriteAccessor< DataVecDeriv > f = vf;
    helper::ReadAccessor< DataVecCoord > x = vx;
    helper::ReadAccessor< DataVecDeriv > v = vv;

    const MassVector &vertexMass= vertexMassInfo.getValue();

    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    // velocity-based stuff
    core::objectmodel::BaseContext::SpatialVector vframe = this->getContext()->getVelocityInWorld();
    core::objectmodel::BaseContext::Vec3 aframe = this->getContext()->getVelocityBasedLinearAccelerationInWorld() ;

    // project back to local frame
    vframe = this->getContext()->getPositionInWorld() / vframe;
    aframe = this->getContext()->getPositionInWorld().backProjectVector( aframe );

    // add weight and inertia force
    if(this->m_separateGravity.getValue())
        for (unsigned int i=0; i<x.size(); ++i)
            f[i] += massLumpingCoeff + core::behavior::inertiaForce(vframe,aframe,vertexMass[i] * massLumpingCoeff ,x[i],v[i]);
    else for (unsigned int i=0; i<x.size(); ++i)
            f[i] += theGravity * vertexMass[i] * massLumpingCoeff + core::behavior::inertiaForce(vframe,aframe,vertexMass[i] * massLumpingCoeff ,x[i],v[i]);
}
#else
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addForce(const core::MechanicalParams*, DataVecDeriv& vf, const DataVecCoord& , const DataVecDeriv& )
{

    //if gravity was added separately (in solver's "solve" method), then nothing to do here
    if(this->m_separateGravity.getValue())
        return ;

    helper::WriteAccessor< DataVecDeriv > f = vf;

    const MassVector &vertexMass= vertexMassInfo.getValue();

    // gravity
    defaulttype::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    // add weight and inertia force
    for (unsigned int i=0; i<f.size(); ++i)
        f[i] += theGravity * vertexMass[i] * massLumpingCoeff;
}
#endif


template <class DataTypes, class MassType>
SReal MeshMatrixMass<DataTypes, MassType>::getKineticEnergy( const core::MechanicalParams*, const DataVecDeriv& vv ) const
{
    const MassVector &vertexMass= vertexMassInfo.getValue();
    const MassVector &edgeMass= edgeMassInfo.getValue();

    helper::ReadAccessor< DataVecDeriv > v = vv;

    unsigned int nbEdges=_topology->getNbEdges();
    unsigned int v0,v1;

    SReal e = 0;

    for (unsigned int i=0; i<v.size(); i++)
    {
        e += dot(v[i],v[i]) * vertexMass[i]; // v[i]*v[i]*masses[i] would be more efficient but less generic
    }
	if (getMassTopologyType()!=TOPOLOGY_BEZIERTETRAHEDRONSET) {
		for (unsigned int i=0; i<nbEdges; ++i)
		{
			v0=_topology->getEdge(i)[0];
			v1=_topology->getEdge(i)[1];

			e += 2*dot(v[v0],v[v1])*edgeMass[i];

		} 
	} else if (bezierTetraGeo ){
//			topology::BezierDegreeType degree=bezierTetraGeo->getTopologyContainer()->getDegree();
//			size_t nbControlPoints=(degree+1)*(degree+2)*(degree+3)/6;
		size_t nbControlPoints;
		typedef typename topology::BezierTetrahedronSetGeometryAlgorithms<DataTypes>::VecPointID VecPointID;
		size_t nbTetras=_topology->getNbTetrahedra();
#ifdef NDEBUG
			assert(tetrahedronMassInfo.size()==(nbControlPoints*(nbControlPoints+1)/2));
#endif
			// go through the mass stored in each tetrahedron element
			size_t rank=0;
			// loop over each tetrahedron of size nbControlPoints*nbControlPoints
			for (size_t i=0; i<nbTetras; i++) {
				
				/// get the global index of each control point in the tetrahedron
				const VecPointID &indexArray=bezierTetraGeo->getTopologyContainer()->getGlobalIndexArrayOfBezierPoints(i) ;
				nbControlPoints=indexArray.size();
				// get the mass matrix in the tetrahedron
//				const MassVector &mv=tetrahedronMassInfo.getValue()[i];
				const MassVector &mv=getBezierTetrahedronMassVector(i);
			//	MassVector mv;
				// loop over each entry in the mass matrix of size nbControlPoints*(nbControlPoints+1)/2
				for (size_t j=0; j<nbControlPoints; ++j) {
					v0 = indexArray[j];
					for (size_t k=j; k<nbControlPoints; ++k,++rank) {
						v1 = indexArray[k];
						
						if (k>j) {
							e += 2*dot(v[v0],v[v1])*mv[rank];
						} else 
							e += dot(v[v0],v[v1])*mv[rank];
					}
				}
			}
		}


    return e/2;
}

template <class DataTypes, class MassType>
 const typename  MeshMatrixMass<DataTypes, MassType>::MassVector &
	 MeshMatrixMass<DataTypes, MassType>::getBezierTetrahedronMassVector(const size_t i) const {
		 return tetrahedronMassInfo.getValue()[i];
 }

template <class DataTypes, class MassType>
SReal MeshMatrixMass<DataTypes, MassType>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx) const
{
    const MassVector &vertexMass= vertexMassInfo.getValue();

    helper::ReadAccessor< DataVecCoord > x = vx;

    SReal e = 0;
    // gravity
    defaulttype::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set ( theGravity, g[0], g[1], g[2]);

    for (unsigned int i=0; i<x.size(); i++)
        e -= dot(theGravity,x[i])*vertexMass[i] * massLumpingCoeff;

    return e;
}


// does nothing by default, need to be specialized in .cpp
template <class DataTypes, class MassType>
defaulttype::Vector6 MeshMatrixMass<DataTypes, MassType>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& /*vx*/, const DataVecDeriv& /*vv*/  ) const
{
    return defaulttype::Vector6();
}



template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v)
{
    if(this->mstate && mparams)
    {
        VecDeriv& v = *d_v.beginEdit();

        // gravity
        defaulttype::Vec3d g ( this->getContext()->getGravity() );
        Deriv theGravity;
        DataTypes::set ( theGravity, g[0], g[1], g[2]);
        Deriv hg = theGravity * (typename DataTypes::Real)(mparams->dt());

        for (unsigned int i=0; i<v.size(); i++)
            v[i] += hg;
        d_v.endEdit();
    }

}




template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    const MassVector &vertexMass= vertexMassInfo.getValue();
    const MassVector &edgeMass= edgeMassInfo.getValue();

    size_t nbEdges=_topology->getNbEdges();
    size_t v0,v1;

    const int N = defaulttype::DataTypeInfo<Deriv>::size();
    AddMToMatrixFunctor<Deriv,MassType> calc;
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = r.matrix;
    Real mFactor = (Real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());

    if((int)mat->colSize() != (_topology->getNbPoints()*N) || (int)mat->rowSize() != (_topology->getNbPoints()*N))
    {
        serr<<"Wrong size of the input Matrix: need resize in addMToMatrix function."<<sendl;
        mat->resize(_topology->getNbPoints()*N,_topology->getNbPoints()*N);
    }

    SReal massTotal=0.0;

    if(this->lumping.getValue())
    {
        for (size_t i=0; i<vertexMass.size(); i++)
        {
            calc(r.matrix, vertexMass[i] * massLumpingCoeff, r.offset + N*i, mFactor);
            massTotal += vertexMass[i] * massLumpingCoeff;
        }

        if(printMass.getValue() && (this->getContext()->getTime()==0.0))
            std::cout<<"Total Mass = "<<massTotal<<std::endl;

        if(printMass.getValue())
        {
            std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
            sofa::helper::vector<double>& graph_error = graph["Mass variations"];
            graph_error.push_back(massTotal);

            f_graph.endEdit();
        }
    }


    else
    {
		if (getMassTopologyType()!=TOPOLOGY_BEZIERTETRAHEDRONSET) {
			for (size_t i=0; i<vertexMass.size(); i++)
			{
				calc(r.matrix, vertexMass[i], r.offset + N*i, mFactor);
				massTotal += vertexMass[i];
			}


			for (size_t j=0; j<nbEdges; ++j)
			{
				v0=_topology->getEdge(j)[0];
				v1=_topology->getEdge(j)[1];

				calc(r.matrix, edgeMass[j], r.offset + N*v0, r.offset + N*v1, mFactor);
				calc(r.matrix, edgeMass[j], r.offset + N*v1, r.offset + N*v0, mFactor);

				massTotal += 2*edgeMass[j];
			}
		} else if (bezierTetraGeo ){
//			topology::BezierDegreeType degree=bezierTetraGeo->getTopologyContainer()->getDegree();
//			size_t nbControlPoints=(degree+1)*(degree+2)*(degree+3)/6;
			size_t nbControlPoints;
			typedef typename topology::BezierTetrahedronSetGeometryAlgorithms<DataTypes>::VecPointID VecPointID;
			size_t nbTetras=_topology->getNbTetrahedra();
#ifdef NDEBUG
			assert(tetrahedronMassInfo.size()==(nbControlPoints*(nbControlPoints+1)/2));
#endif
			// go through the mass stored in each tetrahedron element
			size_t rank=0;
			// loop over each tetrahedron of size nbControlPoints*nbControlPoints
			for (size_t i=0; i<nbTetras; i++) {
				/// get the global index of each control point in the tetrahedron
				const VecPointID &indexArray=bezierTetraGeo->getTopologyContainer()->getGlobalIndexArrayOfBezierPoints(i) ;
				nbControlPoints=indexArray.size();
				// get the mass matrix in the tetrahedron
//				MassVector &mv=tetrahedronMassInfo[i];
				const MassVector &mv=getBezierTetrahedronMassVector(i);
				// loop over each entry in the mass matrix of size nbControlPoints*(nbControlPoints+1)/2
				for (size_t j=0; j<nbControlPoints; ++j) {
					v0 = indexArray[j];
					for (size_t k=j; k<nbControlPoints; ++k,++rank) {
						v1 = indexArray[k];
						calc(r.matrix, mv[rank], r.offset + N*v0, r.offset + N*v1, mFactor);
						
						if (k>j) {
							calc(r.matrix, mv[rank], r.offset + N*v1, r.offset + N*v0, mFactor);
							massTotal += 2*mv[rank];
						} else 
							massTotal += mv[rank];
					}
				}
			}
		}
        if(printMass.getValue() && (this->getContext()->getTime()==0.0))
            std::cout<<"Total Mass  = "<<massTotal<<std::endl;

        if(printMass.getValue())
        {
            std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
            sofa::helper::vector<double>& graph_error = graph["Mass variations"];
            graph_error.push_back(massTotal+0.000001);

            f_graph.endEdit();
        }

    }


}





template <class DataTypes, class MassType>
SReal MeshMatrixMass<DataTypes, MassType>::getElementMass(unsigned int index) const
{
    const MassVector &vertexMass= vertexMassInfo.getValue();
    SReal mass = vertexMass[index] * massLumpingCoeff;

    return mass;
}



//TODO: special case for Rigid Mass
template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const
{
    static const defaulttype::BaseMatrix::Index dimension = (defaulttype::BaseMatrix::Index) defaulttype::DataTypeInfo<Deriv>::size();
    if (m->rowSize() != dimension || m->colSize() != dimension) m->resize(dimension,dimension);

    m->clear();
    AddMToMatrixFunctor<Deriv,MassType>()(m, vertexMassInfo.getValue()[index] * massLumpingCoeff, 0, 1);
}

template <class DataTypes, class MassType>
void MeshMatrixMass<DataTypes, MassType>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels()) return;

    const MassVector &vertexMass= vertexMassInfo.getValue();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Coord gravityCenter;
    Real totalMass=0.0;

	std::vector<  defaulttype::Vector3 > points;
	for (unsigned int i=0; i<x.size(); i++)
	{
		defaulttype::Vector3 p;
		p = DataTypes::getCPos(x[i]);

		points.push_back(p);
		gravityCenter += x[i]*vertexMass[i]*massLumpingCoeff;
		totalMass += vertexMass[i]*massLumpingCoeff;
	}
 


    vparams->drawTool()->drawPoints(points, 2, defaulttype::Vec<4,float>(1,1,1,1));

    if(showCenterOfGravity.getValue())
    {
        glBegin (GL_LINES);
        glColor4f (1,1,0,1);
        glPointSize(5);
        gravityCenter /= totalMass;
        for(unsigned int i=0 ; i<Coord::spatial_dimensions ; i++)
        {
            Coord v;
            v[i] = showAxisSize.getValue();
            helper::gl::glVertexT(gravityCenter-v);
            helper::gl::glVertexT(gravityCenter+v);
        }
        glEnd();
    }
#endif /* SOFA_NO_OPENGL */
}


} // namespace mass

} // namespace component

} // namespace sofa

#endif
