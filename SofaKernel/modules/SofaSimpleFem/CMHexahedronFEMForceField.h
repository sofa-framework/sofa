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
#ifndef SOFA_COMPONENT_FORCEFIELD_CMHEXAHEDRONFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_CMHEXAHEDRONFEMFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/VolumeTopologyContainer.h>
#include <SofaBaseTopology/CMTopologyData.inl>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/behavior/BaseRotationFinder.h>
#include <sofa/helper/decompose.h>
#include <sofa/core/behavior/RotationMatrix.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa
{

namespace component
{

namespace cm_forcefield
{


template<class DataTypes>
class CMHexahedronFEMForceField;

template<class DataTypes>
class CMHexahedronFEMForceFieldInternalData
{
public:
	typedef CMHexahedronFEMForceField<DataTypes> Main;
	void initPtrData(Main * m)
	{
		m->_gatherPt.beginEdit()->setNames(1," ");
		m->_gatherPt.endEdit();

		m->_gatherBsize.beginEdit()->setNames(1," ");
		m->_gatherBsize.endEdit();
	}
};

/** Compute Finite Element forces based on hexahedral elements.
*
* Corotational hexahedron from
* @Article{NMPCPF05,
*   author       = "Nesme, Matthieu and Marchal, Maud and Promayon, Emmanuel and Chabanas, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
*   title        = "Physically Realistic Interactive Simulation for Biological Soft Tissues",
*   journal      = "Recent Research Developments in Biomechanics",
*   volume       = "2",
*   year         = "2005",
*   keywords     = "surgical simulation physical animation truth cube",
*   url          = "http://www-evasion.imag.fr/Publications/2005/NMPCPF05"
* }
*
* WARNING: indices ordering is different than in topology node
*
*     Y  7---------6
*     ^ /         /|
*     |/    Z    / |
*     3----^----2  |
*     |   /     |  |
*     |  4------|--5
*     | /       | /
*     |/        |/
*     0---------1-->X
*/
template<class DataTypes>
class CMHexahedronFEMForceField : virtual public core::behavior::ForceField<DataTypes>, public sofa::core::behavior::BaseRotationFinder
{
public:
	SOFA_CLASS(SOFA_TEMPLATE(CMHexahedronFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

	typedef typename core::behavior::ForceField<DataTypes> InheritForceField;
	typedef typename DataTypes::VecCoord VecCoord;
	typedef typename DataTypes::VecDeriv VecDeriv;
	typedef VecCoord Vector;
	typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Deriv Deriv;
	typedef typename Coord::value_type Real;
	typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
	typedef core::objectmodel::Data<VecCoord> DataVecCoord;
	typedef helper::ReadAccessor< Data< VecCoord > > RDataRefVecCoord;
	typedef helper::WriteAccessor< Data< VecDeriv > > WDataRefVecDeriv;

	using VolumeTopology = sofa::component::topology::VolumeTopologyContainer;
	using BaseVertex = core::topology::MapTopology::Vertex;
	using Vertex = VolumeTopology::Vertex;
	using Volume = VolumeTopology::Volume;
	using VecElement = VolumeTopology::SeqHexahedra;

	template<typename T>
	using VolumeAttribute = typename VolumeTopology::Topology::template VolumeAttribute<T>;

	enum
	{
		LARGE = 0,   ///< Symbol of mean large displacements tetrahedron solver (frame = edges mean on the 3 directions)
		POLAR = 1,   ///< Symbol of polar displacements tetrahedron solver
		SMALL = 2,
	};

protected:

	typedef defaulttype::Vec<24, Real> Displacement;		///< the displacement vector

	typedef defaulttype::Mat<6, 6, Real> MaterialStiffness;	///< the matrix of material stiffness
	using VecMaterialStiffness = VolumeAttribute<MaterialStiffness> ; ///< a vector of material stiffness matrices
	//typedef helper::vector<MaterialStiffness> VecMaterialStiffness;         ///< a vector of material stiffness matrices
	VecMaterialStiffness _materialsStiffnesses;					///< the material stiffness matrices vector

	typedef defaulttype::Mat<24, 24, Real> ElementStiffness;
	using VecElementStiffness = VolumeAttribute<ElementStiffness>;
	//typedef helper::vector<ElementStiffness> VecElementStiffness;
	Data<VecElementStiffness> _elementStiffnesses;

	typedef defaulttype::Mat<3, 3, Real> Mat33;


	typedef std::pair<int,Real> Col_Value;
	typedef helper::vector< Col_Value > CompressedValue;
	typedef helper::vector< CompressedValue > CompressedMatrix;
	CompressedMatrix _stiffnesses;
	SReal m_potentialEnergy;

	typedef unsigned int Index;
	VolumeTopology* _mesh;

	Data< VecCoord > _initialPoints; ///< the intial positions of the points


	defaulttype::Mat<8,3,int> _coef; ///< coef of each vertices to compute the strain stress matrix

	CMHexahedronFEMForceFieldInternalData<DataTypes> *data;
	friend class CMHexahedronFEMForceFieldInternalData<DataTypes>;

public:
	typedef Mat33 Transformation; ///< matrix for rigid transformations like rotations

	int method;
	Data<std::string> f_method; ///< the computation method of the displacements
	Data<Real> f_poissonRatio;
	Data<Real> f_youngModulus;
	Data<bool> f_updateStiffnessMatrix;
	Data<bool> f_assembling;
	Data< sofa::helper::OptionsGroup > _gatherPt; //use in GPU version
	Data< sofa::helper::OptionsGroup > _gatherBsize; //use in GPU version
	Data<bool> f_drawing;
	Data<Real> f_drawPercentageOffset;
	bool needUpdateTopology;

protected:
	CMHexahedronFEMForceField();

public:
	virtual void init();
	virtual void reinit();

	void setPoissonRatio(Real val) { this->f_poissonRatio.setValue(val); }
	void setYoungModulus(Real val) { this->f_youngModulus.setValue(val); }
	void setMethod(int val)
	{
		method = val;
		switch(val)
		{
		case POLAR: f_method.setValue("polar"); break;
		case SMALL: f_method.setValue("small"); break;
		default   : f_method.setValue("large");
		};
	}

	void setUpdateStiffnessMatrix(bool val) { this->f_updateStiffnessMatrix.setValue(val); }
	void setComputeGlobalMatrix(bool val) { this->f_assembling.setValue(val); }



	virtual void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

	virtual void addDForce (const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx);

	virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
	{
		serr << "Get potentialEnergy not implemented" << sendl;
		return 0.0;
	}

	// Make other overloaded version of getPotentialEnergy() to show up in subclass.
	using InheritForceField::getPotentialEnergy;
	// getPotentialEnergy is implemented for polar method
	virtual SReal getPotentialEnergy(const core::MechanicalParams*) const;

	const Transformation& getElementRotation(const Volume w) { return _rotations[w]; }

	void getNodeRotation(Transformation& R, Vertex nodeIdx)
	{
		//core::topology::MapTopology::HexahedraAroundVertex liste_hexa = _mesh->getHexahedraAroundVertex(nodeIdx);

		R[0][0] = R[1][1] = R[2][2] = 1.0 ;
		R[0][1] = R[0][2] = R[1][0] = R[1][2] = R[2][0] = R[2][1] = 0.0 ;
		/*
		unsigned int numHexa = liste_hexa.size();
		for (unsigned int ti=0; ti < numHexa; ti++)
		{
			Transformation R0t;
			R0t.transpose(_initialrotations[Volume(liste_hexa[ti])]);
			Transformation Rcur = getElementRotation(Volume(liste_hexa[ti]));
			R += Rcur * R0t;
		}
		*/
		unsigned int numHexa = 0;
		_mesh->foreach_incident_volume(nodeIdx, [&](Volume w){
			Transformation R0t;
			R0t.transpose(_initialrotations[w]);
			Transformation Rcur = getElementRotation(w);
			R += Rcur * R0t;
			++numHexa;
		});

		// on "moyenne"
		R[0][0] = R[0][0]/numHexa ; R[0][1] = R[0][1]/numHexa ; R[0][2] = R[0][2]/numHexa ;
		R[1][0] = R[1][0]/numHexa ; R[1][1] = R[1][1]/numHexa ; R[1][2] = R[1][2]/numHexa ;
		R[2][0] = R[2][0]/numHexa ; R[2][1] = R[2][1]/numHexa ; R[2][2] = R[2][2]/numHexa ;

		defaulttype::Mat<3,3,Real> Rmoy;
		helper::Decompose<Real>::polarDecomposition( R, Rmoy );

		R = Rmoy;
	}

	void getRotations(defaulttype::BaseMatrix * rotations, int offset = 0)
	{
		//unsigned int nbdof = this->mstate->getSize();

		if (component::linearsolver::RotationMatrix<float> * diag = dynamic_cast<component::linearsolver::RotationMatrix<float> *>(rotations))
		{
			Transformation R;
			_mesh->foreach_cell([&](Vertex v)
			{
				getNodeRotation(R,v);
				unsigned int e = _mesh->get_dof(v);
				for(int j=0; j<3; j++)
				{
					for(int i=0; i<3; i++)
					{
						diag->getVector()[e*9 + j*3 + i] = (float)R[j][i];
					}
				}
			});
		}
		else if (component::linearsolver::RotationMatrix<double> * diag = dynamic_cast<component::linearsolver::RotationMatrix<double> *>(rotations))
		{
			Transformation R;
			_mesh->foreach_cell([&](Vertex v)
			{
				getNodeRotation(R,v);
				unsigned int e = _mesh->get_dof(v);

				for(int j=0; j<3; j++)
				{
					for(int i=0; i<3; i++)
					{
						diag->getVector()[e*9 + j*3 + i] = R[j][i];
					}
				}
			});
		}
		else
		{
			_mesh->foreach_cell([&](Vertex v)
			{
				Transformation t;
				getNodeRotation(t,v);
				unsigned int i = _mesh->get_dof(v);

				int e = offset+i*3;
				rotations->set(e+0,e+0,t[0][0]); rotations->set(e+0,e+1,t[0][1]); rotations->set(e+0,e+2,t[0][2]);
				rotations->set(e+1,e+0,t[1][0]); rotations->set(e+1,e+1,t[1][1]); rotations->set(e+1,e+2,t[1][2]);
				rotations->set(e+2,e+0,t[2][0]); rotations->set(e+2,e+1,t[2][1]); rotations->set(e+2,e+2,t[2][2]);
			});
		}
	}

	void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix);


	void computeBBox(const core::ExecParams* params, bool onlyVisible);

	void draw(const core::visual::VisualParams* vparams);

	void handleTopologyChange() { needUpdateTopology = true; }


protected:


	inline const VecElement *getIndexedElements() { return & (_mesh->getHexahedra()); }

	virtual void computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const helper::fixed_array<Coord,8> &nodes, const Index elementIndice, double stiffnessFactor=1.0);
	Mat33 integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  );

	void computeMaterialStiffness(Volume w);

	void computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K );


	////////////// large displacements method
	VolumeAttribute<helper::fixed_array<Coord,8>> _rotatedInitialElements;   ///< The initials positions in its frame
	VolumeAttribute<Transformation> _rotations;
	VolumeAttribute<Transformation> _initialrotations;
	//helper::vector<helper::fixed_array<Coord,8> > _rotatedInitialElements;   ///< The initials positions in its frame
	//helper::vector<Transformation> _rotations;
	//helper::vector<Transformation> _initialrotations;
	void initLarge(Volume w);
	void computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey);
	virtual void accumulateForceLarge( WDataRefVecDeriv &f, RDataRefVecCoord &p, Volume w);

	////////////// polar decomposition method
	void initPolar(Volume w);
	void computeRotationPolar( Transformation &r, defaulttype::Vec<8,Coord> &nodes);
	virtual void accumulateForcePolar( WDataRefVecDeriv &f, RDataRefVecCoord &p, Volume w);

	////////////// small decomposition method
	void initSmall(Volume i);
	virtual void accumulateForceSmall( WDataRefVecDeriv &f, RDataRefVecCoord &p, Volume w);

	bool _alreadyInit;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_CMHEXAHEDRONFEMFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_SIMPLE_FEM_API CMHexahedronFEMForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_SIMPLE_FEM_API CMHexahedronFEMForceField<defaulttype::Vec3fTypes>;
#endif

#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CMHEXAHEDRONFEMFORCEFIELD_H
