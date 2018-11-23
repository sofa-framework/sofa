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

#ifndef SOFACOMBINATORIALMAPS_FORCEFIELD_CMTRIANGULARFEMFORCEFIELD_INL_
#define SOFACOMBINATORIALMAPS_FORCEFIELD_CMTRIANGULARFEMFORCEFIELD_INL_

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaCombinatorialMaps/ForceFields/CMTriangularFEMForceField.h>

#include <sofa/core/visual/VisualParams.h>
#include <SofaOpenglVisual/OglColorMap.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/gl.h>

#include <SofaCombinatorialMaps/BaseTopology/CMTopologyData.inl>

#include <sofa/helper/system/thread/debug.h>
#include <newmat/newmat.h>
#include <newmat/newmatap.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <algorithm>
#include <limits>

#include <cgogn/geometry/types/plane_3d.h>

#ifdef DEBUG_TRIANGLEFEM
    #define DEBUG_TRIANGLEFEM_MSG true
#else
    #define DEBUG_TRIANGLEFEM_MSG false
#endif



namespace sofa
{

namespace component
{

namespace cm_forcefield
{

// --------------------------------------------------------------------------------------
// ---  Topology Creation/Destruction functions
// --------------------------------------------------------------------------------------

template< class DataTypes>
void CMTriangularFEMForceField<DataTypes>::TRQSTriangleHandler::applyCreateFunction(
		TriangleInformation &,
		const core::topology::CMapTopology::Face &t,
		const sofa::helper::vector<unsigned int> &,
		const sofa::helper::vector<double> &)
{
	/*
	const Face f(t);
	if (ff)
    {

		const VecCoord X0 = ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

		TriangleInformation& info = triangleInfo[f];

		auto& t = ff->_topology->get_dofs(f);
		for (int i=0; i<3; ++i) info.dofs[i] = t[i];

		switch(ff->method)
        {
        case SMALL :
			ff->initSmall(f, X0, info);
			ff->computeMaterialStiffness(info);
            break;

        case LARGE :
			ff->initLarge(f, X0, info);
			ff->computeMaterialStiffness(info);
            break;
		}
	}*/
}


// --------------------------------------------------------------------------------------
// --- constructor
// --------------------------------------------------------------------------------------
template <class DataTypes>
CMTriangularFEMForceField<DataTypes>::CMTriangularFEMForceField() :
//    triangleInfo(initData(&triangleInfo, "triangleInfo", "Internal triangle data"))
//    , vertexInfo(initData(&vertexInfo, "vertexInfo", "Internal point data"))
//    , edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
	_topology(NULL)
    , method(LARGE)
    , f_method(initData(&f_method,std::string("large"),"method","large: large displacements, small: small displacements"))
//, f_poisson(initData(&f_poisson,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
//, f_young(initData(&f_young,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , f_poisson(initData(&f_poisson,helper::vector<Real>(1,static_cast<Real>(0.45)),"poissonRatio","Poisson ratio in Hooke's law (vector)"))
    , f_young(initData(&f_young,helper::vector<Real>(1,static_cast<Real>(1000.0)),"youngModulus","Young modulus in Hooke's law (vector)"))
    , f_damping(initData(&f_damping,(Real)0.,"damping","Ratio damping/stiffness"))
    , m_rotatedInitialElements(initData(&m_rotatedInitialElements,"rotatedInitialElements","Flag activating rendering of stress directions within each triangle"))
    , m_initialTransformation(initData(&m_initialTransformation,"initialTransformation","Flag activating rendering of stress directions within each triangle"))
    , f_fracturable(initData(&f_fracturable,false,"fracturable","the forcefield computes the next fracturable Edge"))
    , hosfordExponant(initData(&hosfordExponant, (Real)1.0, "hosfordExponant","Exponant in the Hosford yield criteria"))
    , criteriaValue(initData(&criteriaValue, (Real)1e15, "criteriaValue","Fracturable threshold used to draw fracturable triangles"))
    , showStressValue(initData(&showStressValue,false,"showStressValue","Flag activating rendering of stress values as a color in each triangle"))
    , showStressVector(initData(&showStressVector,false,"showStressVector","Flag activating rendering of stress directions within each triangle"))
    , showFracturableTriangles(initData(&showFracturableTriangles,false,"showFracturableTriangles","Flag activating rendering of triangles to fracture"))
    , f_computePrincipalStress(initData(&f_computePrincipalStress,false,"computePrincipalStress","Compute principal stress for each triangle"))
	//, faces_cache(initLink("travFaces", "A personalized face traversor"))
	//, mask(initLink("mask", "Traversal mask"))
#ifdef PLOT_CURVE
    , elementID( initData(&elementID, (Real)0, "id","element id to follow for fracture criteria") )
    , f_graphStress( initData(&f_graphStress,"graphMaxStress","Graph of max stress corresponding to the element id") )
    , f_graphCriteria( initData(&f_graphCriteria,"graphCriteria","Graph of the fracture criteria corresponding to the element id") )
    , f_graphOrientation( initData(&f_graphOrientation,"graphOrientation","Graph of the orientation of the principal stress direction corresponding to the element id"))
#endif
{

    _anisotropicMaterial = false;
	//triangleHandler = new TRQSTriangleHandler(this, &triangleInfo);
#ifdef PLOT_CURVE
    f_graphStress.setWidget("graph");
    f_graphCriteria.setWidget("graph");
    f_graphOrientation.setWidget("graph");
#endif

    f_poisson.setRequired(true);
    f_young.setRequired(true);
}


template <class DataTypes>
CMTriangularFEMForceField<DataTypes>::~CMTriangularFEMForceField()
{
	if(triangleHandler) delete triangleHandler;

    cell_traversor.release();
}


// --------------------------------------------------------------------------------------
// --- Initialization stage
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::init()
{
    this->Inherited::init();

#ifdef PLOT_CURVE
    allGraphStress.clear();
    allGraphCriteria.clear();
    allGraphOrientation.clear();

    f_graphStress.beginEdit()->clear();
    f_graphStress.endEdit();
    f_graphCriteria.beginEdit()->clear();
    f_graphCriteria.endEdit();
    f_graphOrientation.beginEdit()->clear();
    f_graphOrientation.endEdit();
#endif

	this->getContext()->get(_topology);

	if(_topology == nullptr)
		msg_error("CMTriangularFEMForceField") << "no topology";

	cell_traversor = cgogn::make_unique<FilteredQuickTraversor>(_topology->getMap());

//    if (!mask.get())
//        cell_traversor->build<Face>();
//    else
//    {
//        cell_traversor->build<Face>();
//        cell_traversor->set_filter<Face>(std::move(std::ref(*(mask.get()))));
//    }


	m_rotatedInitialElements = _topology->add_attribute<RotatedInitialElements, Face>("CMTriangularFEMForceField_RotatedInitialElements");
	m_initialTransformation = _topology->add_attribute<Transformation, Face>("CMTriangularFEMForceField_Transformation");

	// Create specific handler for TriangleData
//	auto& attributeV = *(vertexInfo.beginEdit());
	vertexInfo = _topology->add_attribute<VertexInformation, Vertex>("CMTriangularFEMForceField_VertexInformation");
//	vertexInfo.endEdit();

//	auto& attributeE = *(edgeInfo.beginEdit());
	edgeInfo = _topology->add_attribute<EdgeInformation, Edge>("CMTriangularFEMForceField_EdgeInformation");
//	edgeInfo.endEdit();

//	auto& attributeF = *(triangleInfo.beginEdit());
	triangleInfo = _topology->add_attribute<TriangleInformation, Face>("CMTriangularFEMForceField_TriangleInformation");
//	triangleInfo.endEdit();

    if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;

	if (_topology->nb_cells<SurfaceTopology::Face::ORBIT>() == 0u)
    {
        serr << "ERROR(TriangularFEMForceField): object must have a Triangular Set Topology."<<sendl;
        return;
    }

    lastFracturedEdgeIndex = -1;

    reinit();
}

// --------------------------------------------------------------------------------------
// --- Re-initialization (called when we change a parameter through the GUI)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::reinit()
{
	if (f_method.getValue() == "small")
		method = SMALL;
	else if (f_method.getValue() == "large")
		method = LARGE;


#ifdef PLOT_CURVE
	std::map<std::string, sofa::helper::vector<double> > &stress = *(f_graphStress.beginEdit());
	stress.clear();
	if (allGraphStress.size() > elementID.getValue())
		stress = allGraphStress[elementID.getValue()];
	f_graphStress.endEdit();

	std::map<std::string, sofa::helper::vector<double> > &criteria = *(f_graphCriteria.beginEdit());
	criteria.clear();
	if (allGraphCriteria.size() > elementID.getValue())
		criteria = allGraphCriteria[elementID.getValue()];
	f_graphCriteria.endEdit();

	std::map<std::string, sofa::helper::vector<double> > &orientation = *(f_graphOrientation.beginEdit());
	orientation.clear();
	if (allGraphOrientation.size() > elementID.getValue())
		orientation = allGraphOrientation[elementID.getValue()];
	f_graphOrientation.endEdit();
#endif
}

// --------------------------------------------------------------------------------------
// --- AddForce and AddDForce methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
	VecDeriv& f1 = *f.beginEdit();
	const VecCoord& x1 = x.getValue();

	f1.resize(x1.size());

	if(f_damping.getValue() != 0)
	{
		if(method == SMALL)
		{
			_topology->foreach_cell([&](Face face)
			{
				accumulateForceSmall( f1, x1, face);
				accumulateDampingSmall( f1, face);
			}, *cell_traversor);
		}
		else
		{
			_topology->foreach_cell([&](Face face)
			{
				accumulateForceLarge( f1, x1, face);
				accumulateDampingLarge( f1, face);
			}, *cell_traversor);
		}
	}
	else
	{
		if (method==SMALL)
		{
			_topology->foreach_cell([&](Face face)
			{
				accumulateForceSmall( f1, x1, face);
			}, *cell_traversor);
		}
		else
		{
			_topology->foreach_cell([&](Face face)
			{
				accumulateForceLarge( f1, x1, face);
			}, *cell_traversor);
		}
	}
	f.endEdit();

	if (f_computePrincipalStress.getValue())
	{
		_topology->foreach_cell([&](Face face)
		{
			TriangleInformation& info  = triangleInfo[face.dart];
			computePrincipalStress(info);
		}, *cell_traversor);
	}
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
	VecDeriv& df1 = *df.beginEdit();
	const VecDeriv& dx1 = dx.getValue();
	Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

	Real h=1;
	df1.resize(dx1.size());

	if (method == SMALL)
		applyStiffnessSmall( df1, h, dx1, kFactor );
	else
		applyStiffnessLarge( df1, h, dx1, kFactor );

	df.endEdit();
}

// --------------------------------------------------------------------------------------
// --- Get/Set methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
SReal CMTriangularFEMForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* x */) const
{
	msg_error() << "TriangularFEMForceField::getPotentialEnergy-not-implemented !!!";
	return 0;
}


// --------------------------------------------------------------------------------------
// --- Display methods
// --------------------------------------------------------------------------------------
template<class DataTypes>
void CMTriangularFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
	if (!vparams->displayFlags().getShowForceFields())
		return;

	if (vparams->displayFlags().getShowWireFrame())
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

	glDisable(GL_LIGHTING);
	if (!f_fracturable.getValue() && !this->showFracturableTriangles.getValue())
	{
		glBegin(GL_TRIANGLES);
		_topology->foreach_cell([&](Face face)
		{
			const auto& dofs = _topology->get_dofs(face);

			Index a = dofs[0];
			Index b = dofs[1];
			Index c = dofs[2];

			glColor4f(0,1,0,1);
			helper::gl::glVertexT(x[a]);
			glColor4f(0,0.5,0.5,1);
			helper::gl::glVertexT(x[b]);
			glColor4f(0,0,1,1);
			helper::gl::glVertexT(x[c]);
		}, *cell_traversor);
		glEnd();
	}

	if (showStressVector.getValue() || showStressValue.getValue() || showFracturableTriangles.getValue())
	{
		_topology->foreach_cell([&](Face face)
		{
			computePrincipalStress(triangleInfo[face]);
		}, *cell_traversor);
	}

	if (showStressVector.getValue())
	{
		glColor4f(1,0,1,1);
		glBegin(GL_LINES);
		_topology->foreach_cell([&](Face face)
		{
			const auto& dofs = _topology->get_dofs(face);

			Index a = dofs[0];
			Index b = dofs[1];
			Index c = dofs[2];

			Coord center = (x[a]+x[b]+x[c])/3;
			Coord d = triangleInfo[face].principalStressDirection*2.5; //was 0.25
			helper::gl::glVertexT(center);
			helper::gl::glVertexT(center+d);
		}, *cell_traversor);
		glEnd();
	}

	if (showStressValue.getValue())
	{
		double minStress = numeric_limits<double>::max();
		double maxStress = 0.0;
		_topology->foreach_cell([&](Vertex vertex)
		{
			double averageStress = 0.0;
			double sumArea = 0.0;

			_topology->foreach_incident_face(vertex, [&](Face face)
			{
				if (triangleInfo[face].area)
				{
					averageStress+= ( fabs(triangleInfo[face].maxStress) * triangleInfo[face].area);
					sumArea += triangleInfo[face].area;
				}
			});
			if (sumArea)
				averageStress /= sumArea;

			vertexInfo[vertex].stress = averageStress;

			if (averageStress < minStress )
				minStress = averageStress;
			if (averageStress > maxStress)
				maxStress = averageStress;
		}, *cell_traversor);

		helper::ColorMap::evaluator<double> evalColor = helper::ColorMap::getDefault()->getEvaluator(minStress, maxStress);
		glBegin(GL_TRIANGLES);
		_topology->foreach_cell([&](Face face)
		{
			const auto& dofs = _topology->get_dofs(face);

			Index a = dofs[0];
			Index b = dofs[1];
			Index c = dofs[2];

			glColor4fv(evalColor(vertexInfo[a].stress).ptr());
			helper::gl::glVertexT(x[a]);
			glColor4fv(evalColor(vertexInfo[b].stress).ptr());
			helper::gl::glVertexT(x[b]);
			glColor4fv(evalColor(vertexInfo[c].stress).ptr());
			helper::gl::glVertexT(x[c]);
		}, *cell_traversor);
		glEnd();
	}

	if (showFracturableTriangles.getValue())
	{
		Real maxDifference = numeric_limits<Real>::min();
		Real minDifference = numeric_limits<Real>::max();
		_topology->foreach_cell([&](Face face)
		{
			if (triangleInfo[face].differenceToCriteria > 0)
			{
				if (triangleInfo[face].differenceToCriteria > maxDifference)
					maxDifference = triangleInfo[face].differenceToCriteria;

				if (triangleInfo[face].differenceToCriteria < minDifference)
					minDifference = triangleInfo[face].differenceToCriteria;
			}
		}, *cell_traversor);

		glBegin(GL_TRIANGLES);
		_topology->foreach_cell([&](Face face)
		{
			if (triangleInfo[face].differenceToCriteria > 0)
			{
				glColor4d( 0.4 + 0.4 * (triangleInfo[face].differenceToCriteria - minDifference ) /  (maxDifference - minDifference) , 0.0 , 0.0, 0.5);

				const auto& dofs = _topology->get_dofs(face);

				Index a = dofs[0];
				Index b = dofs[1];
				Index c = dofs[2];

				helper::gl::glVertexT(x[a]);
				helper::gl::glVertexT(x[b]);
				helper::gl::glVertexT(x[c]);
			}
		}, *cell_traversor);
		glEnd();
	}

	if (vparams->displayFlags().getShowWireFrame())
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif /* SOFA_NO_OPENGL */
}


// --------------------------------------------------------------------------------------
// --- Get Fracture Criteria
// --------------------------------------------------------------------------------------
template<class DataTypes>
int CMTriangularFEMForceField<DataTypes>::getFracturedEdge()
{
	if (f_fracturable.getValue())
	{
		Edge fractured;
		_topology->foreach_cell([&] (Edge edge) -> bool
		{
			if (edgeInfo[edge].fracturable)
			{
				fractured = edge;
				return false;
			}

			return true;
		});

		return fractured.dart.index;
	}

	return -1;
}

template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::getFractureCriteria(TriangleInformation& info, Deriv& direction, Real& value)
{
	//TODO(dmarchal 2017-05-03) Who wrote this todo ? When will you fix this ? In one year I remove this one.
	/// @todo evaluate the criteria on the current position instead of relying on the computations during the force evaluation (based on the previous position)

	computePrincipalStress(info);
	direction = info.principalStressDirection;
	value = fabs(info.maxStress);
	if (value < 0)
	{
		direction.clear();
		value = 0;
	}
}

// ----------------------------------------------------------------------------------------------------------------------------------------
// ---	Compute value of stress along a given direction (typically the fiber direction and transverse direction in anisotropic materials)
// ----------------------------------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeStressAlongDirection(Real &stress_along_dir, const TriangleInformation& info, const Coord &dir, const defaulttype::Vec<3,Real> &stress)
{
	defaulttype::Mat<3,3,Real> R, Rt;

	// transform 'dir' into local coordinates
	R = info.rotation;
	Rt.transpose(R);
	Coord dir_local = Rt * dir;
	dir_local[2] = 0; // project direction
	dir_local.normalize();

	// compute stress along specified direction 'dir'
	Real cos_theta = dir_local[0];
	Real sin_theta = dir_local[1];
	stress_along_dir = stress[0]*cos_theta*cos_theta + stress[1]*sin_theta*sin_theta + stress[2]*2*cos_theta*sin_theta;
}

template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeStressAlongDirection(Real &stress_along_dir, Face face, const Coord &dir)
{
	defaulttype::Vec<3,Real> stress;
	this->computeStress(stress, face);
	this->computeStressAlongDirection(stress_along_dir, triangleInfo[face], dir, stress);
}

template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeStressAcrossDirection(Real &stress_across_dir, Face face, const Coord &dir, const defaulttype::Vec<3,Real> &stress)
{
	const auto& dofs = _topology->get_dofs(face);

	Index a = dofs[0];
	Index b = dofs[1];
	Index c = dofs[2];

	const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
	Coord n = cross(x[b]-x[a],x[c]-x[a]);
	Coord dir_t = cross(dir,n);
	this->computeStressAlongDirection(stress_across_dir, triangleInfo[face], dir_t, stress);
}

template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeStressAcrossDirection(Real &stress_across_dir, Face face, const Coord &dir)
{
	const auto& dofs = _topology->get_dofs(face);

	Index a = dofs[0];
	Index b = dofs[1];
	Index c = dofs[2];
	const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
	Coord n = cross(x[b]-x[a],x[c]-x[a]);
	Coord dir_t = cross(dir,n);
	this->computeStressAlongDirection(stress_across_dir, face, dir_t);
}

/// Compute current stress
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeStress(defaulttype::Vec<3,Real> &stress, Face face)
{
	Displacement D;
	StrainDisplacement J;
	defaulttype::Vec<3,Real> strain;
	Transformation R_0_2, R_2_0;
	const VecCoord& p = this->mstate->read(core::ConstVecCoordId::position())->getValue();
	const auto& dofs = _topology->get_dofs(face);

	Index a = dofs[0];
	Index b = dofs[1];
	Index c = dofs[2];

	if (method == SMALL)
	{
		// classic linear elastic method
		R_0_2.identity();
		computeDisplacementSmall(D, face, p);
		if (_anisotropicMaterial)
			computeStrainDisplacement(J, face, Coord(0,0,0), (p[b]-p[a]), (p[c]-p[a]));
		else
			J = triangleInfo[face].strainDisplacementMatrix;
		computeStrain(strain, J, D);
		computeStress(stress, triangleInfo[face].materialMatrix, strain);
	}
	else
	{
		// co-rotational method
		// first, compute rotation matrix into co-rotational frame
		computeRotationLarge( R_0_2, p, face);

		// then compute displacement in this frame
		computeDisplacementLarge(D, face, R_0_2, p);

		// and compute postions of a, b, c in the co-rotational frame
		Coord A = Coord(0, 0, 0);
		Coord B = R_0_2 * (p[b]-p[a]);
		Coord C = R_0_2 * (p[c]-p[a]);

		computeStrainDisplacement(J, face, A, B, C);
		computeStrain(strain, J, D);
		computeStress(stress, triangleInfo[face].materialMatrix, strain);
	}
	// store newly computed values for next time
	R_2_0.transpose(R_0_2);
	triangleInfo[face].strainDisplacementMatrix = J;
	triangleInfo[face].rotation = R_2_0;
	triangleInfo[face].strain = strain;
	triangleInfo[face].stress = stress;
}

// --------------------------------------------------------------------------------------
// --- Get the rotation of node
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::getRotation(Transformation& R, Vertex vertex)
{
	Transformation r;
	unsigned int numNeiTri = 0;
	_topology->foreach_incident_face(vertex, [&](Face face)
	{
		TriangleInformation& tinfo = triangleInfo[face];
		Transformation r01,r21;
		r01=tinfo.initialTransformation;
		r21=tinfo.rotation*r01;
		r+=r21;
		++numNeiTri;
	});

	R=r/static_cast<Real>(numNeiTri);

	//orthogonalization
	Coord ex,ey,ez;
	for(int i=0; i<3; i++)
	{
		ex[i]=R[0][i];
		ey[i]=R[1][i];
	}
	ex.normalize();
	ey.normalize();

	ez=cross(ex,ey);
	ez.normalize();

	ey=cross(ez,ex);
	ey.normalize();

	for(int i=0; i<3; i++)
	{
		R[0][i]=ex[i];
		R[1][i]=ey[i];
		R[2][i]=ez[i];
	}
}

// --------------------------------------------------------------------------------------
// --- Get the rotation of node
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::getRotations()
{
	//reset the rotation matrix
	_topology->foreach_cell([&](Vertex vertex)
	{
		vertexInfo[vertex].rotation.clear();
	}, *cell_traversor);

	//add the rotation matrix
	_topology->foreach_cell([&](Face face)
	{
		TriangleInformation& tinfo = triangleInfo[face];
		Transformation r01,r21;
		r01=tinfo.initialTransformation;
		r21=tinfo.rotation*r01;

		_topology->foreach_incident_vertex(face, [&](Vertex vertex)
		{
			vertexInfo[vertex].rotation+=r21;
		});
	}, *cell_traversor);

	//averaging the rotation matrix
	_topology->foreach_cell([&](Vertex vertex)
	{
		VertexInformation& vinfo = vertexInfo[vertex];
		int numNeiTri=0;
		_topology->foreach_incident_face(vertex, [&](Face ) {++numNeiTri;	});
		vinfo.rotation/=static_cast<Real>(numNeiTri);

		//orthogonalization
		Coord ex,ey,ez;
		for(int i=0; i<3; i++)
		{
			ex[i]=vinfo.rotation[0][i];
			ey[i]=vinfo.rotation[1][i];
		}
		ex.normalize();
		ey.normalize();

		ez=cross(ex,ey);
		ez.normalize();

		ey=cross(ez,ex);
		ey.normalize();

		for(int i=0; i<3; i++)
		{
			vinfo.rotation[0][i]=ex[i];
			vinfo.rotation[1][i]=ey[i];
			vinfo.rotation[2][i]=ez[i];
		}
	}, *cell_traversor);
}

// ---------------------------------------------------------------------------------------------------------------
// ---	Compute displacement vector D as the difference between current current position 'p' and initial position
// ---------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeDisplacementSmall(Displacement &D, Face face, const VecCoord &p)
{
	const auto& dofs = _topology->get_dofs(face);

	Index a = dofs[0];
	Index b = dofs[1];
	Index c = dofs[2];

	//Coord deforme_a = Coord(0,0,0);
	Coord deforme_b = p[b]-p[a];
	Coord deforme_c = p[c]-p[a];

	D[0] = 0;
	D[1] = 0;
	D[2] = triangleInfo[face].rotatedInitialElements[1][0] - deforme_b[0];
	D[3] = triangleInfo[face].rotatedInitialElements[1][1] - deforme_b[1];
	D[4] = triangleInfo[face].rotatedInitialElements[2][0] - deforme_c[0];
	D[5] = triangleInfo[face].rotatedInitialElements[2][1] - deforme_c[1];
}

// -------------------------------------------------------------------------------------------------------------
// --- Compute displacement vector D as the difference between current current position 'p' and initial position
// --- expressed in the co-rotational frame of reference
// -------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeDisplacementLarge(Displacement &D, Face face, const Transformation &R_0_2, const VecCoord &p)
{
	const auto& dofs = _topology->get_dofs(face);

	Index a = dofs[0];
	Index b = dofs[1];
	Index c = dofs[2];

	// positions of the deformed and displaced triangle in its frame
	Coord deforme_b = R_0_2 * (p[b]-p[a]);
	Coord deforme_c = R_0_2 * (p[c]-p[a]);

	// displacements
	D[0] = 0;
	D[1] = 0;
	D[2] = triangleInfo[face].rotatedInitialElements[1][0] - deforme_b[0];
	D[3] = 0;
	D[4] = triangleInfo[face].rotatedInitialElements[2][0] - deforme_c[0];
	D[5] = triangleInfo[face].rotatedInitialElements[2][1] - deforme_c[1];

	if ( D[2] != D[2] || D[4] != D[4] || D[5] != D[5])
	{
		msg_info() << "computeDisplacementLarge :: deforme_b = " <<  deforme_b << msgendl
				   << "computeDisplacementLarge :: deforme_c = " <<  deforme_c << msgendl
				   << "computeDisplacementLarge :: R_0_2 = " <<  R_0_2 << msgendl;
	}
}

// ------------------------------------------------------------------------------------------------------------
// --- Compute the strain-displacement matrix where (a, b, c) are the coordinates of the 3 nodes of a triangle
// ------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeStrainDisplacement(StrainDisplacement &J, Face face, Coord a, Coord b, Coord c )
{
	Real determinant;

	if (method == SMALL)
	{
		Coord ab_cross_ac = cross(b-a, c-a);
		determinant = ab_cross_ac.norm();
		triangleInfo[face].area = determinant*0.5f;

		Real x13 = (a[0]-c[0]) / determinant;
		Real x21 = (b[0]-a[0]) / determinant;
		Real x32 = (c[0]-b[0]) / determinant;
		Real y12 = (a[1]-b[1]) / determinant;
		Real y23 = (b[1]-c[1]) / determinant;
		Real y31 = (c[1]-a[1]) / determinant;

		J[0][0] = y23;
		J[0][1] = 0;
		J[0][2] = x32;

		J[1][0] = 0;
		J[1][1] = x32;
		J[1][2] = y23;

		J[2][0] = y31;
		J[2][1] = 0;
		J[2][2] = x13;

		J[3][0] = 0;
		J[3][1] = x13;
		J[3][2] = y31;

		J[4][0] = y12;
		J[4][1] = 0;
		J[4][2] = x21;

		J[5][0] = 0;
		J[5][1] = x21;
		J[5][2] = y12;
	}
	else
	{
		determinant = b[0] * c[1];
		triangleInfo[face].area = determinant*0.5f;

		Real x13 = -c[0] / determinant; // since a=(0,0)
		Real x21 = b[0] / determinant; // since a=(0,0)
		Real x32 = (c[0]-b[0]) / determinant;
		Real y12 = 0;	// since a=(0,0) and b[1] = 0
		Real y23 = -c[1] / determinant; // since a=(0,0) and b[1] = 0
		Real y31 = c[1] / determinant; // since a=(0,0)

		J[0][0] = y23; // -cy   / det
		J[0][1] = 0;   // 0
		J[0][2] = x32; // cx-bx / det

		J[1][0] = 0;   // 0
		J[1][1] = x32; // cx-bx / det
		J[1][2] = y23; // -cy   / det

		J[2][0] = y31; // cy    / det
		J[2][1] = 0;   // 0
		J[2][2] = x13; // -cx   / det

		J[3][0] = 0;   // 0
		J[3][1] = x13; // -cx   / det
		J[3][2] = y31; // cy    / det

		J[4][0] = y12; // 0
		J[4][1] = 0;   // 0
		J[4][2] = x21; // bx    / det

		J[5][0] = 0;   // 0
		J[5][1] = x21; // bx    / det
		J[5][2] = y12; // 0
	}
}

// --------------------------------------------------------------------------------------------------------
// --- Stiffness = K = J*D*Jt
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeStiffness(StrainDisplacement &J, Stiffness &K, MaterialStiffness &D)
{
	defaulttype::Mat<3,6,Real> Jt;
	Jt.transpose(J);
	K=J*D*Jt;
}

// --------------------------------------------------------------------------------------------------------
// --- Strain = StrainDisplacement * Displacement = JtD = Bd
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeStrain(defaulttype::Vec<3,Real> &strain, const StrainDisplacement &J, const Displacement &D)
{
	defaulttype::Mat<3,6,Real> Jt;
	Jt.transpose(J);

	if (_anisotropicMaterial || method == SMALL)
	{
		strain = Jt * D;
	}
	else
	{
		strain[0] = Jt[0][0] * D[0] + /* Jt[0][1] * Depl[1] + */ Jt[0][2] * D[2] /* + Jt[0][3] * Depl[3] + Jt[0][4] * Depl[4] + Jt[0][5] * Depl[5] */ ;
		strain[1] = /* Jt[1][0] * Depl[0] + */ Jt[1][1] * D[1] + /* Jt[1][2] * Depl[2] + */ Jt[1][3] * D[3] + /* Jt[1][4] * Depl[4] + */ Jt[1][5] * D[5];
		strain[2] = Jt[2][0] * D[0] + Jt[2][1] * D[1] + Jt[2][2] * D[2] +	Jt[2][3] * D[3] + Jt[2][4] * D[4] /* + Jt[2][5] * Depl[5] */ ;
	}
}

// --------------------------------------------------------------------------------------------------------
// --- Stress = K * Strain = KJtD = KBd
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeStress(defaulttype::Vec<3,Real> &stress, MaterialStiffness &K, defaulttype::Vec<3,Real> &strain)
{
	if (_anisotropicMaterial || method == SMALL)
	{
		stress = K * strain;
	}
	else
	{
		// Optimisations: The following values are 0 (per computeMaterialStiffnesses )
		// K[0][2]  K[1][2]  K[2][0] K[2][1]
		stress[0] = K[0][0] * strain[0] + K[0][1] * strain[1] + K[0][2] * strain[2];
		stress[1] = K[1][0] * strain[0] + K[1][1] * strain[1] + K[1][2] * strain[2];
		stress[2] = K[2][0] * strain[0] + K[2][1] * strain[1] + K[2][2] * strain[2];
	}
}

// --------------------------------------------------------------------------------------
// ---	Compute F = J * stress;
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeForce(Displacement &F, Face face, const VecCoord &p)
{
	Displacement D;
	StrainDisplacement J;
	Stiffness K;
	defaulttype::Vec<3,Real> strain;
	defaulttype::Vec<3,Real> stress;
	Transformation R_0_2, R_2_0;

	const auto& dofs = _topology->get_dofs(face);

	Index a = dofs[0];
	Index b = dofs[1];
	Index c = dofs[2];

	if (method == SMALL)
	{
		// classic linear elastic method
		computeDisplacementSmall(D, face, p);
		computeStrainDisplacement(J, face, Coord(0,0,0), (p[b]-p[a]), (p[c]-p[a]));
		computeStrain(strain, J, D);
		computeStress(stress, triangleInfo[face].materialMatrix, strain);
		F = J * stress * triangleInfo[face].area;

		// store newly computed values for next time
		triangleInfo[face].strainDisplacementMatrix = J;
		triangleInfo[face].strain = strain;
		triangleInfo[face].stress = stress;
	}
	else
	{
		// co-rotational method
		// first, compute rotation matrix into co-rotational frame
		computeRotationLarge( R_0_2, p, face);

		// then compute displacement in this frame
		computeDisplacementLarge(D, face, R_0_2, p);

		// and compute postions of a, b, c in the co-rotational frame
		Coord A = Coord(0, 0, 0);
		Coord B = R_0_2 * (p[b]-p[a]);
		Coord C = R_0_2 * (p[c]-p[a]);

		if (_anisotropicMaterial)
			computeStrainDisplacement(J, face, A, B, C);
		else
			J = triangleInfo[face].strainDisplacementMatrix;
		computeStrain(strain, J, D);
		computeStress(stress, triangleInfo[face].materialMatrix, strain);
		computeStiffness(J,K,triangleInfo[face].materialMatrix);

		// Compute F = J * stress;
		// Optimisations: The following values are 0 (per computeStrainDisplacement )
		// J[0][1] J[1][0] J[2][1] J[3][0] J[4][0] J[4][1] J[5][0] J[5][2]

		F[0] = J[0][0] * stress[0] + /* J[0][1] * KJtD[1] + */ J[0][2] * stress[2];
		F[1] = /* J[1][0] * KJtD[0] + */ J[1][1] * stress[1] + J[1][2] * stress[2];
		F[2] = J[2][0] * stress[0] + /* J[2][1] * KJtD[1] + */ J[2][2] * stress[2];
		F[3] = /* J[3][0] * KJtD[0] + */ J[3][1] * stress[1] + J[3][2] * stress[2];
		F[4] = /* J[4][0] * KJtD[0] + J[4][1] * KJtD[1] + */ J[4][2] * stress[2];
		F[5] = /* J[5][0] * KJtD[0] + */ J[5][1] * stress[1] /* + J[5][2] * KJtD[2] */ ;

		// Since J has been "normalized" we need to multiply the force F by the area of the triangle to get the correct force
		F *= triangleInfo[face].area;

		// store newly computed values for next time
		R_2_0.transpose(R_0_2);
		triangleInfo[face].strainDisplacementMatrix = J;
		triangleInfo[face].rotation = R_2_0;
		triangleInfo[face].strain = strain;
		triangleInfo[face].stress = stress;
		triangleInfo[face].stiffness = K;
	}
}

// --------------------------------------------------------------------------------------
// ---	Compute direction of maximum strain (strain = JtD = BD)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computePrincipalStrain(TriangleInformation& info, defaulttype::Vec<3,Real> &strain )
{
	NEWMAT::SymmetricMatrix e(2);
	e = 0.0;

	NEWMAT::DiagonalMatrix D(2);
	D = 0.0;

	NEWMAT::Matrix V(2,2);
	V = 0.0;

	e(1,1) = strain[0];
	e(1,2) = strain[2];
	e(2,1) = strain[2];
	e(2,2) = strain[1];

	NEWMAT::Jacobi(e, D, V);

	Coord v((Real)V(1,1), (Real)V(2,1), 0.0);
	v.normalize();


	info.maxStrain = (Real)D(1,1);

	info.principalStrainDirection = info.rotation * Coord(v[0], v[1], v[2]);
	info.principalStrainDirection *= info.maxStrain/100.0;
}

// --------------------------------------------------------------------------------------
// ---	Compute direction of maximum stress (stress = KJtD = KBD)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computePrincipalStress(TriangleInformation& info)
{
	defaulttype::Vec<3,Real>& stress = info.stress;
	NEWMAT::SymmetricMatrix e(2);
	e = 0.0;

	NEWMAT::DiagonalMatrix D(2);
	D = 0.0;

	NEWMAT::Matrix V(2,2);
	V = 0.0;

	//voigt notation to symmetric matrix
	e(1,1) = stress[0];
	e(1,2) = stress[2];
	e(2,1) = stress[2];
	e(2,2) = stress[1];

	//compute eigenvalues and eigenvectors
	NEWMAT::Jacobi(e, D, V);

	//get the index of the biggest eigenvalue in absolute value
	unsigned int biggestIndex = 0;
	if (fabs(D(1,1)) > fabs(D(2,2)))
		biggestIndex = 1;
	else
		biggestIndex = 2;

	//get the eigenvectors corresponding to the biggest eigenvalue
	//note : according to newmat doc => The elements of D are sorted in ascending order, The eigenvectors are returned as the columns of V
	Coord direction((Real)V(1,biggestIndex), (Real)V(2,biggestIndex), 0.0);
	direction.normalize();

	//Hosford yield criterion
	//for plane stress : 1/2 * ( |S_1|^n + |S_2|^n) + 1/2 * |S_1 - S_2|^n = S_y^n
	//with S_i the principal stresses, n is a material-dependent exponent and S_y is the yield stress in uniaxial tension/compression
	double n = this->hosfordExponant.getValue();
	info.differenceToCriteria = (Real)
			pow(0.5 * (pow((double)fabs(D(1,1)), n) +  pow((double)fabs(D(2,2)), n) + pow((double)fabs(D(1,1) - D(2,2)),n)), 1.0/ n) - this->criteriaValue.getValue();

	//max stress is the highest eigenvalue
	info.maxStress = fabs((Real)D(biggestIndex,biggestIndex));

	//the principal stress direction is the eigenvector corresponding to the highest eigenvalue
	Coord principalStressDir = info.rotation * direction;//need to rotate to be in global frame instead of local
	principalStressDir *= info.maxStress/100.0;


	//make an average of the n1 and n2 last stress direction to smooth it and avoid discontinuities
	unsigned int n2 = 30;
	unsigned int n1 = 10;
	info.lastNStressDirection.push_back(principalStressDir);

	//remove useless data
	if (info.lastNStressDirection.size() > n2)
	{
		for ( unsigned int i = 0 ; i  < info.lastNStressDirection.size() - n2 ; i++)
			info.lastNStressDirection.erase(info.lastNStressDirection.begin()+i);
	}

	//make the average
	Coord averageVector2(0.0,0.0,0.0);
	Coord averageVector1(0.0,0.0,0.0);
	for (unsigned int i = 0 ; i < info.lastNStressDirection.size() ; i++)
	{
		averageVector2 = info.lastNStressDirection[i] + averageVector2;
		if (i == n1)
			averageVector1 = averageVector2 / n1;
	}
	if (info.lastNStressDirection.size())
		averageVector2 /=  info.lastNStressDirection.size();

	info.principalStressDirection = averageVector2;

#ifdef PLOT_CURVE
	Coord direction2((Real)V(1,D.Ncols() +1 - biggestIndex), (Real)V(2,D.Ncols() +1 - biggestIndex), 0.0);
	direction2.normalize();

	Coord principalStressDir2 = triangleInf[elementIndex].rotation * direction2;//need to rotate to be in global frame instead of local
	principalStressDir2 *= fabs((Real)D(D.Ncols() +1 - biggestIndex,D.Ncols() +1 - biggestIndex))/100.0;

	//compute an angle between the principal stress direction and the x-axis
	Real orientation2 = dot( averageVector2, Coord(1.0, 0.0, 0.0));
	Real orientation1 = dot( averageVector1, Coord(1.0, 0.0, 0.0));
	Real orientation0 = dot( principalStressDir, Coord(1.0, 0.0, 0.0));
	Real orientationSecond = dot( principalStressDir2, Coord(1.0, 0.0, 0.0));

	/* store the values which are plot*/
	if (allGraphStress.size() <= elementIndex)
		allGraphStress.resize(elementIndex+1);
	if (allGraphCriteria.size() <= elementIndex)
		allGraphCriteria.resize(elementIndex+1);
	if (allGraphOrientation.size() <= elementIndex)
		allGraphOrientation.resize(elementIndex+1);

	std::map<std::string, sofa::helper::vector<double> > &stressMap = allGraphStress[elementIndex];
	std::map<std::string, sofa::helper::vector<double> > &criteriaMap = allGraphCriteria[elementIndex];
	std::map<std::string, sofa::helper::vector<double> > &orientationMap = allGraphOrientation[elementIndex];

	stressMap["first stress eigenvalue"].push_back((double)(triangleInf[elementIndex].maxStress));
	stressMap["second stress eigenvalue"].push_back((double)(fabs(D(1,1))));

	criteriaMap["fracture criteria"].push_back((double)(triangleInf[elementIndex].differenceToCriteria));

	orientationMap["principal stress direction orientation with 30-average"].push_back((double)(acos(orientation2) * 180 / 3.14159265));
	orientationMap["principal stress direction orientation with 10-average"].push_back((double)(acos(orientation1) * 180 / 3.14159265));
	orientationMap["principal stress direction orientation with no-average"].push_back((double)(acos(orientation0) * 180 / 3.14159265));
	orientationMap["second stress direction orientation with no-average"].push_back((double)(acos(orientationSecond) * 180 / 3.14159265));


	//save values in graphs
	if (elementIndex == elementID.getValue())
	{
		std::map < std::string, sofa::helper::vector<double> >& graphStress = *f_graphStress.beginEdit();
		sofa::helper::vector<double>& graph_maxStress1 = graphStress["first stress eigenvalue"];
		graph_maxStress1.push_back((double)(triangleInf[elementIndex].maxStress));
		sofa::helper::vector<double>& graph_maxStress2 = graphStress["second stress eigenvalue"];
		graph_maxStress2.push_back((double)(fabs(D(1,1))));
		f_graphStress.endEdit();

		std::map < std::string, sofa::helper::vector<double> >& graphCriteria = *f_graphCriteria.beginEdit();
		sofa::helper::vector<double>& graph_criteria = graphCriteria["fracture criteria"];
		graph_criteria.push_back((double)(triangleInf[elementIndex].differenceToCriteria));
		f_graphCriteria.endEdit();

		std::map < std::string, sofa::helper::vector<double> >& graphOrientation = *f_graphOrientation.beginEdit();
		sofa::helper::vector<double>& graph_orientation2 = graphOrientation["principal stress direction orientation with 30-average"];
		graph_orientation2.push_back((double)(acos(orientation2) * 180 / 3.14159265));
		sofa::helper::vector<double>& graph_orientation1 = graphOrientation["principal stress direction orientation with 10-average"];
		graph_orientation1.push_back((double)(acos(orientation1) * 180 / 3.14159265));
		sofa::helper::vector<double>& graph_orientation0 = graphOrientation["principal stress direction orientation with no-average"];
		graph_orientation0.push_back((double)(acos(orientation0) * 180 / 3.14159265));
		sofa::helper::vector<double>& graph_orientationSecond = graphOrientation["second stress direction orientation with no-average"];
		graph_orientationSecond.push_back((double)(acos(orientationSecond) * 180 / 3.14159265));
		f_graphOrientation.endEdit();
	}
#endif

}

// --------------------------------------------------------------------------------------
// ---	Apply Stiffness
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::applyStiffness( VecCoord& v, Real h, const VecCoord& x, const SReal &kFactor )
{
	if (method == SMALL)
		applyStiffnessSmall( v, h, x, kFactor );
	else
		applyStiffnessLarge( v, h, x, kFactor );
}

// --------------------------------------------------------------------------------------
// ---	Compute material stiffness
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeMaterialStiffness(TriangleInformation& info)
{
	const helper::vector<Real> & youngArray = f_young.getValue();
	const helper::vector<Real> & poissonArray = f_poisson.getValue();

	Real y = youngArray[0] ;
	Real p = poissonArray[0];

	info.materialMatrix[0][0] = 1;
	info.materialMatrix[0][1] = p;//poissonArray[i];//f_poisson.getValue();
	info.materialMatrix[0][2] = 0;
	info.materialMatrix[1][0] = p;//poissonArray[i];//f_poisson.getValue();
	info.materialMatrix[1][1] = 1;
	info.materialMatrix[1][2] = 0;
	info.materialMatrix[2][0] = 0;
	info.materialMatrix[2][1] = 0;
	info.materialMatrix[2][2] = (1.0f - p) * 0.5f;//poissonArray[i]);

	info.materialMatrix *= (y / (1.0f - p * p));
}

// --------------------------------------------------------------------------------------
// --- Store the initial position of the nodes
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::initSmall(Face f, const VecCoord& X0, TriangleInformation& info)
{
	if(DEBUG_TRIANGLEFEM_MSG)
		dmsg_info() << "Entering initSmall" ;

	info.initialTransformation.identity();

	if (m_rotatedInitialElements.isSet())
		info.rotatedInitialElements = m_rotatedInitialElements.getValue()[f];
	else
	{
		info.rotatedInitialElements[0] = (X0)[info.dofs[0]] - (X0)[info.dofs[0]]; // always (0,0,0)
		info.rotatedInitialElements[1] = (X0)[info.dofs[1]] - (X0)[info.dofs[0]];
		info.rotatedInitialElements[2] = (X0)[info.dofs[2]] - (X0)[info.dofs[0]];
	}

	computeStrainDisplacement(info.strainDisplacementMatrix, f, info.rotatedInitialElements[0], info.rotatedInitialElements[1], info.rotatedInitialElements[2]);
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::accumulateForceSmall( VecCoord &f, const VecCoord &p, Face face)
{
	Displacement F;

	const auto& dofs = _topology->get_dofs(face);

	Index a = dofs[0];
	Index b = dofs[1];
	Index c = dofs[2];

	// compute force on element
	computeForce(F, face, p);

	f[a] += Coord( F[0], F[1], 0);
	f[b] += Coord( F[2], F[3], 0);
	f[c] += Coord( F[4], F[5], 0);
}

// --------------------------------------------------------------------------------------
// --- Accumulate functions
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::accumulateDampingSmall(VecCoord&, Face )
{
	if(DEBUG_TRIANGLEFEM_MSG)
		dmsg_info() << "TriangularFEMForceField::accumulateDampingSmall" ;
}

// --------------------------------------------------------------------------------------
// --- Apply functions
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::applyStiffnessSmall(VecCoord &v, Real h, const VecCoord &x, const SReal &kFactor)
{
	if(DEBUG_TRIANGLEFEM_MSG)
		dmsg_info() << "Entering in applyStiffnessSmall." ;

	defaulttype::Mat<6,3,Real> J;
	defaulttype::Vec<3,Real> strain, stress;
	Displacement D, F;

	_topology->foreach_cell([&](Face face)
	{
		const auto& dofs = _topology->get_dofs(face);

		Index a = dofs[0];
		Index b = dofs[1];
		Index c = dofs[2];

		D[0] = x[a][0];
		D[1] = x[a][1];

		D[2] = x[b][0];
		D[3] = x[b][1];

		D[4] = x[c][0];
		D[5] = x[c][1];

		J = triangleInfo[face].strainDisplacementMatrix;
		computeStrain(strain, J, D);
		computeStress(stress, triangleInfo[face].materialMatrix, strain);
		F = J * stress * triangleInfo[face].area;

		v[a] += (Coord(-h*F[0], -h*F[1], 0)) * kFactor;
		v[b] += (Coord(-h*F[2], -h*F[3], 0)) * kFactor;
		v[c] += (Coord(-h*F[4], -h*F[5], 0)) * kFactor;
	}, *cell_traversor);
}

// --------------------------------------------------------------------------------------
// --- Store the initial position of the nodes
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::initLarge(Face f, const VecCoord& X0, TriangleInformation& info)
{
    if(DEBUG_TRIANGLEFEM_MSG)
        dmsg_info() << "Entering initLarge" ;

    if (m_initialTransformation.isSet() && m_rotatedInitialElements.isSet())
    {
        Transformation R_0_1;
		R_0_1 = m_initialTransformation.getValue()[f];
		info.initialTransformation = R_0_1;
		info.rotatedInitialElements = m_rotatedInitialElements.getValue()[f];
    }
    else
    {
        // Rotation matrix (initial triangle/world)
        // first vector on first edge
        // second vector in the plane of the two first edges
        // third vector orthogonal to first and second
        Transformation R_0_1;

		computeRotationLarge( R_0_1, X0, f);

		info.initialTransformation = R_0_1;

		info.rotatedInitialElements[0] = R_0_1 * (X0)[info.dofs[0]] - (X0)[info.dofs[0]];
		info.rotatedInitialElements[1] = R_0_1 * (X0)[info.dofs[1]] - (X0)[info.dofs[0]];
		info.rotatedInitialElements[2] = R_0_1 * (X0)[info.dofs[2]] - (X0)[info.dofs[0]];
    }

	computeStrainDisplacement(info.strainDisplacementMatrix, f, info.rotatedInitialElements[0], info.rotatedInitialElements[1], info.rotatedInitialElements[2]);
}

// --------------------------------------------------------------------------------------
// --- Computation methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::computeRotationLarge( Transformation &r, const VecCoord &p, Face face)
{

    if(DEBUG_TRIANGLEFEM_MSG)
        dmsg_info() << "Entering in computeRotationLarge.";

	const auto& dofs = _topology->get_dofs(face);

	Index a = dofs[0];
	Index b = dofs[1];
	Index c = dofs[2];


    /// first vector on first edge
    /// second vector in the plane of the two first edges
    /// third vector orthogonal to first and second
    Coord edgex = p[b] - p[a];
    edgex.normalize();

    Coord edgey = p[c] - p[a];
    edgey.normalize();

    Coord edgez;
    edgez = cross(edgex, edgey);
    edgez.normalize();

    edgey = cross(edgez, edgex);
    edgey.normalize();

    r[0][0] = edgex[0];
    r[0][1] = edgex[1];
    r[0][2] = edgex[2];
    r[1][0] = edgey[0];
    r[1][1] = edgey[1];
    r[1][2] = edgey[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];

    if ( r[0][0]!=r[0][0])
    {
        msg_info() << "computeRotationLarge::edgex " << edgex << msgendl
                   << "computeRotationLarge::edgey " << edgey << msgendl
                   << "computeRotationLarge::edgez " << edgez << msgendl
                   << "computeRotationLarge::pa " << p[a] << msgendl
                   << "computeRotationLarge::pb " << p[b] << msgendl
                   << "computeRotationLarge::pc " <<  p[c] << msgendl;
    }
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::accumulateForceLarge(VecCoord &f, const VecCoord &p, Face face )
{
	if(DEBUG_TRIANGLEFEM_MSG)
		dmsg_info() << "TriangularFEMForceField::accumulateForceLarge" ;

	Displacement F;

	const auto& dofs = _topology->get_dofs(face);

	Index a = dofs[0];
	Index b = dofs[1];
	Index c = dofs[2];

	// compute force on element (in the co-rotational space)
	computeForce( F, face, p);

	// transform force back into global ref. frame
	f[a] += triangleInfo[face].rotation * Coord(F[0], F[1], 0);
	f[b] += triangleInfo[face].rotation * Coord(F[2], F[3], 0);
	f[c] += triangleInfo[face].rotation * Coord(F[4], F[5], 0);
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::accumulateDampingLarge(VecCoord &, Face )
{
	if(DEBUG_TRIANGLEFEM_MSG)
		dmsg_info() << "TriangularFEMForceField::accumulateDampingLarge" ;
}

// --------------------------------------------------------------------------------------
// ---
// --------------------------------------------------------------------------------------
template <class DataTypes>
void CMTriangularFEMForceField<DataTypes>::applyStiffnessLarge(VecCoord &v, Real h, const VecCoord &x, const SReal &kFactor)
{

	if(DEBUG_TRIANGLEFEM_MSG)
		msg_info() << "Entering applyStiffnessLarge" ;

	defaulttype::Mat<6,3,Real> J;
	defaulttype::Vec<3,Real> strain, stress;
	MaterialStiffness K;
	Displacement D;
	Coord x_2;

	_topology->foreach_cell([&](Face face)
	{
		const auto& dofs = _topology->get_dofs(face);

		Index a = dofs[0];
		Index b = dofs[1];
		Index c = dofs[2];

		Transformation R_0_2;
		R_0_2.transpose(triangleInfo[face].rotation);

		VecCoord disp;
		disp.resize(3);

		x_2 = R_0_2 * x[a];
		disp[0] = x_2;

		D[0] = x_2[0];
		D[1] = x_2[1];

		x_2 = R_0_2 * x[b];
		disp[1] = x_2;
		D[2] = x_2[0];
		D[3] = x_2[1];

		x_2 = R_0_2 * x[c];
		disp[2] = x_2;
		D[4] = x_2[0];
		D[5] = x_2[1];

		Displacement F;

		K = triangleInfo[face].materialMatrix;
		J = triangleInfo[face].strainDisplacementMatrix;

		computeStrain(strain, J, D);
		computeStress(stress, triangleInfo[face].materialMatrix, strain);

		F[0] = J[0][0] * stress[0] + /* J[0][1] * KJtD[1] + */ J[0][2] * stress[2];
		F[1] = /* J[1][0] * KJtD[0] + */ J[1][1] * stress[1] + J[1][2] * stress[2];
		F[2] = J[2][0] * stress[0] + /* J[2][1] * KJtD[1] + */ J[2][2] * stress[2];
		F[3] = /* J[3][0] * KJtD[0] + */ J[3][1] * stress[1] + J[3][2] * stress[2];
		F[4] = /* J[4][0] * KJtD[0] + J[4][1] * KJtD[1] + */ J[4][2] * stress[2];
		F[5] = /* J[5][0] * KJtD[0] + */ J[5][1] * stress[1] /* + J[5][2] * KJtD[2] */ ;

		F *= triangleInfo[face].area;

		v[a] += (triangleInfo[face].rotation * Coord(-h*F[0], -h*F[1], 0)) * kFactor;
		v[b] += (triangleInfo[face].rotation * Coord(-h*F[2], -h*F[3], 0)) * kFactor;
		v[c] += (triangleInfo[face].rotation * Coord(-h*F[4], -h*F[5], 0)) * kFactor;
	}, *cell_traversor);
}

} // namespace forcefield

} // namespace component

} // namespace sofa


#endif //SOFACOMBINATORIALMAPS_FORCEFIELD_CMTRIANGULARFEMFORCEFIELD_INL_
