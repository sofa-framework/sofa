/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/solidmechanics/fem/hyperelastic/config.h>


#include <sofa/component/solidmechanics/fem/hyperelastic/material/HyperelasticMaterial.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/type/MatSym.h>
#include <sofa/type/trait/Rebind.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologyData.h>


namespace sofa::component::solidmechanics::fem::hyperelastic
{


//***************** Tetrahedron FEM code for several elastic models: StandardTetrahedralFEMForceField*******************************************************************
//********************************** Based on classical discretization : Fi=-Bi^T S V and Kij=Bi^T N Bj +Di^T S Dj **********************************************
//***************************************** where Bi is the strain displacement (6*3 matrix), S SPK tensor N=dS/dC, Di shape vector ************************************
//**************************** Code dependant on HyperelasticMatrialFEM and inherited classes *********************************************************************

/** Compute Finite Element forces based on tetrahedral elements.
*/
template<class DataTypes>
class StandardTetrahedralFEMForceField: public core::behavior::ForceField<DataTypes>
{
  public:
	  SOFA_CLASS(SOFA_TEMPLATE(StandardTetrahedralFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef type::Mat<3,3,Real> Matrix3;
    typedef type::Mat<6,6,Real> Matrix6;
    typedef type::Mat<6,3,Real> Matrix63;
    typedef type::MatSym<3,Real> MatrixSym;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv; 
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord; 

    typedef type::vector<Real> SetParameterArray;
    typedef type::vector<Coord> SetAnisotropyDirectionArray;

    typedef core::topology::BaseMeshTopology::Index Index;
    typedef core::topology::BaseMeshTopology::Tetra Element;
	typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra VecElement;

public :
	
    material::MaterialParameters<DataTypes> globalParameters;

	
	

    /// data structure stored for each tetrahedron
	class TetrahedronRestInformation : public material::StrainInformation<DataTypes>
    {
    public:

      /// rest volume
	  Real restVolume;
      /// current tetrahedron volume
      Real volScale;

      /// shape vector at the rest configuration
	  Coord shapeVector[4];

	  /// fiber direction in rest configuration
      Coord fiberDirection;
	 
	  /// derivatives of J
	  Coord dJ[4];
//	  MatrixSym SPKTensorGeneral;
	  /// deformation gradient = gradPhi
	//  Matrix3 deformationGradient;
	  /// right Cauchy-Green deformation tensor C (gradPhi^T gradPhi) 
	  //Matrix63 matB[4];
	  Real strainEnergy;

      //Tetrahedron Points Indicies for CUDA
      float tetraIndices[4]{};
      //Tetrahedron Edges for CUDA
      float tetraEdges[6]{};

      /// Output stream
      inline friend std::ostream& operator<< ( std::ostream& os, const TetrahedronRestInformation& /*eri*/ ) {  return os;  }
      /// Input stream
      inline friend std::istream& operator>> ( std::istream& in, TetrahedronRestInformation& /*eri*/ ) { return in; }

      TetrahedronRestInformation() : restVolume(0), volScale(0), fiberDirection(), strainEnergy(0) {}

    };
	using tetrahedronRestInfoVector  = type::rebind_to<VecCoord, TetrahedronRestInformation>;
    
	
   /// data structure stored for each edge
   class EdgeInformation
   {
   public:
	   /// store the stiffness edge matrix 
	   Matrix3 DfDx;
       float vertices[2];

	   /// Output stream
	   inline friend std::ostream& operator<< ( std::ostream& os, const EdgeInformation& /*eri*/ ) {  return os;  }
	   /// Input stream
	   inline friend std::istream& operator>> ( std::istream& in, EdgeInformation& /*eri*/ ) { return in; }

     EdgeInformation(): DfDx() { vertices[0]=0.f; vertices[1]=0.f; }
   };
   using edgeInformationVector  = type::rebind_to<VecCoord, EdgeInformation>;

 protected :
   core::topology::BaseMeshTopology* m_topology;
   VecCoord  _initialPoints;	/// the intial positions of the points
   bool updateMatrix;
   bool  _meshSaved ;
   Data<std::string> f_materialName; ///< the name of the material
   Data<SetParameterArray> f_parameterSet; ///< The global parameters specifying the material
   Data<SetAnisotropyDirectionArray> f_anisotropySet; ///< The global directions of anisotropy of the material
   Data<std::string> f_parameterFileName; ///< the name of the file describing the material parameters for all tetrahedra

   /// Link to be set to the topology container in the component graph.
   SingleLink<StandardTetrahedralFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

   
public:

	void setMaterialName(const std::string& name) {
		f_materialName.setValue(name);
	}
    void setparameter(const type::vector<Real>& param) {
		f_parameterSet.setValue(param);
	}
    void setdirection(const type::vector<Coord>& direction) {
		f_anisotropySet.setValue(direction);
	}

protected:
   StandardTetrahedralFEMForceField();
   
   virtual   ~StandardTetrahedralFEMForceField();
public:

  //  virtual void parse(core::objectmodel::BaseObjectDescription* arg);

    void init() override;
    //Used for CUDA implementation
    void initNeighbourhoodPoints();
    void initNeighbourhoodEdges();
    void addKToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal kFact, unsigned int &offset) override;
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    void draw(const core::visual::VisualParams* vparams) override;

  //  type::Mat<3,3,double> getPhi( int );

    /** Method to initialize @sa TetrahedronRestInformation when a new Tetrahedron is created.
    * Will be set as creation callback in the TetrahedronData @sa tetrahedronInfo
    */
    void createTetrahedronRestInformation(Index, TetrahedronRestInformation& t,
        const core::topology::BaseMeshTopology::Tetrahedron&,
        const sofa::type::vector<Index>&,
        const sofa::type::vector<SReal>&);
	
  protected:
    /// the array that describes the complete material energy and its derivatives

    material::HyperelasticMaterial<DataTypes> *myMaterial;

    core::topology::TetrahedronData<tetrahedronRestInfoVector> tetrahedronInfo; ///< Internal tetrahedron data
    core::topology::EdgeData<edgeInformationVector> edgeInfo; ///< Internal edge data


    void testDerivatives();
    void saveMesh( const char *filename );
	
	VecCoord myposition;
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_STANDARDTETRAHEDRALFEMFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API StandardTetrahedralFEMForceField<sofa::defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_STANDARDTETRAHEDRALFEMFORCEFIELD_CPP)


} // namespace sofa::component::solidmechanics::fem::hyperelastic
