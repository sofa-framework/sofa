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
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/helper/map.h>
#include <sofa/helper/ColorMap.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

// corotational tetrahedron from
// @InProceedings{NPF05,
//   author       = "Nesme, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
//   title        = "Efficient, Physically Plausible Finite Elements",
//   booktitle    = "Eurographics (short papers)",
//   month        = "august",
//   year         = "2005",
//   editor       = "J. Dingliana and F. Ganovelli",
//   keywords     = "animation, physical model, elasticity, finite elements",
//   url          = "http://www-evasion.imag.fr/Publications/2005/NPF05"
// }


namespace sofa::component::solidmechanics::fem::elastic
{

/** Compute Finite Element forces based on tetrahedral elements.
 */
template<class DataTypes>
class TetrahedralCorotationalFEMForceField : public BaseLinearElasticityFEMForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TetrahedralCorotationalFEMForceField, DataTypes), SOFA_TEMPLATE(BaseLinearElasticityFEMForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    using Index = sofa::Index;

    enum { SMALL = 0, ///< Symbol of small displacements tetrahedron solver
            LARGE = 1, ///< Symbol of large displacements tetrahedron solver
            POLAR = 2  ///< Symbol of polar displacements tetrahedron solver
         };

    /// @name Per element (tetrahedron) data
    /// @{

    /// Displacement vector (deformation of the 4 corners of a tetrahedron
    typedef type::VecNoInit<12, Real> Displacement;

    /// Material stiffness matrix of a tetrahedron
    typedef type::Mat<6, 6, Real> MaterialStiffness;

    /// Strain-displacement matrix
    typedef type::Mat<12, 6, Real> StrainDisplacementTransposed;

    /// Rigid transformation (rotation) matrix
    typedef type::MatNoInit<3, 3, Real> Transformation;

    /// Stiffness matrix ( = RJKJtRt  with K the Material stiffness matrix, J the strain-displacement matrix, and R the transformation matrix if any )
    typedef type::Mat<12, 12, Real> StiffnessMatrix;

    /// @}

    /// the information stored for each tetrahedron
    class TetrahedronInformation
    {
    public:
        /// material stiffness matrices of each tetrahedron
        MaterialStiffness materialMatrix;
        /// the strain-displacement matrices vector
        StrainDisplacementTransposed strainDisplacementTransposedMatrix;
        /// large displacement method
        type::fixed_array<Coord,4> rotatedInitialElements;
        type::Mat<4, 4, Real> elemShapeFun;
        Transformation rotation;
        /// polar method
        Transformation initialTransformation;

        TetrahedronInformation()
        {
        }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TetrahedronInformation& /*tri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TetrahedronInformation& /*tri*/ )
        {
            return in;
        }
    };

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::vector<TetrahedronInformation> > tetrahedronInfo;

    /// container that stotes all requires information for each tetrahedron
    core::topology::TetrahedronData<sofa::type::vector<TetrahedronInformation> > d_tetrahedronInfo;

    /// @name Full system matrix assembly support
    /// @{

    typedef std::pair<int,Real> Col_Value;
    typedef type::vector< Col_Value > CompressedValue;
    typedef type::vector< CompressedValue > CompressedMatrix;

    CompressedMatrix _stiffnesses;
    /// @}

    SReal m_potentialEnergy;

    sofa::core::topology::BaseMeshTopology* _topology;

public:
    int method;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<std::string> f_method;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> _poissonRatio;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    SOFA_ATTRIBUTE_DISABLED("", "v24.12", "Use d_youngModulus instead") DeprecatedAndRemoved _youngModulus;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<VecReal> _localStiffnessFactor;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> _updateStiffnessMatrix;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> _assembling;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> f_drawing;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::RGBAColor> drawColor1;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::RGBAColor> drawColor2;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::RGBAColor> drawColor3;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::RGBAColor> drawColor4;

    Data<std::string> d_method; ///< "small", "large" (by QR) or "polar" displacements
    Data<VecReal> d_localStiffnessFactor; ///< Allow specification of different stiffness per element. If there are N element and M values are specified, the youngModulus factor for element i would be localStiffnessFactor[i*M/N]
    Data<bool> d_updateStiffnessMatrix;
    Data<bool> d_assembling;
    Data<bool> d_drawing; ///<  draw the forcefield if true
    Data<bool> _displayWholeVolume;
    Data<sofa::type::RGBAColor> d_drawColor1; ///<  draw color for faces 1
    Data<sofa::type::RGBAColor> d_drawColor2; ///<  draw color for faces 2
    Data<sofa::type::RGBAColor> d_drawColor3; ///<  draw color for faces 3
    Data<sofa::type::RGBAColor> d_drawColor4; ///<  draw color for faces 4
    Data<std::map < std::string, sofa::type::vector<double> > > _volumeGraph;
    
    Data<bool> d_computeVonMisesStress;
    Data<type::vector<Real> > d_vonMisesPerElement; ///< von Mises Stress per element
    Data<type::vector<Real> > d_vonMisesPerNode; ///< von Mises Stress per node
    
    sofa::helper::ColorMap* m_vonMisesColorMap;
    Real prevMaxStress = -1.0;

    using Inherit1::l_topology;


protected:
    TetrahedralCorotationalFEMForceField();

    ~TetrahedralCorotationalFEMForceField() override;

public:

    void setMethod(int val) { method = val; }

    void setUpdateStiffnessMatrix(bool val) { this->d_updateStiffnessMatrix.setValue(val); }

    void setComputeGlobalMatrix(bool val) { this->d_assembling.setValue(val); }

    void init() override;
    void reinit() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        msg_warning() << "Method getPotentialEnergy not implemented yet.";
        return 0.0;
    }

    void addKToMatrix(sofa::linearalgebra::BaseMatrix *m, SReal kFactor, unsigned int &offset) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    // Getting the rotation of the vertex by averaing the rotation of neighboring elements
    void getRotation(Transformation& R, Index nodeIdx);
    void getRotations() {}
    void getElementRotation(Transformation& R, Index elementIdx);

    // Getting the stiffness matrix of index i
    void getElementStiffnessMatrix(Real* stiffness, Index nodeIdx);


    void draw(const core::visual::VisualParams* vparams) override;

    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    void computeVonMisesStress();

protected:
    /** Method to create @sa TetrahedronInformation when a new tetrahedron is created.
    * Will be set as creation callback in the TetrahedronData @sa d_tetrahedronInfo
    */
    void createTetrahedronInformation(Index tetrahedronIndex, TetrahedronInformation& tInfo,
        const core::topology::BaseMeshTopology::Tetrahedron& tetra,
        const sofa::type::vector<Index>& ancestors,
        const sofa::type::vector<SReal>& coefs);

    void computeStrainDisplacement( StrainDisplacementTransposed &J, Coord a, Coord b, Coord c, Coord d );
    Real peudo_determinant_for_coef ( const type::Mat<2, 3, Real>&  M );

    void computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacementTransposed &J, const Transformation& Rot );

    void computeMaterialStiffness(int i, Index&a, Index&b, Index&c, Index&d);

    /// overloaded by classes with non-uniform stiffness
    virtual void computeMaterialStiffness(int tetrahedronId, MaterialStiffness& materialMatrix, Index&a, Index&b, Index&c, Index&d, SReal localStiffnessFactor=1);

    void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacementTransposed &J );
    void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacementTransposed &J, SReal fact );

    ////////////// small displacements method
    void initSmall(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceSmall( Vector& f, const Vector & p, Index elementIndex );
    void applyStiffnessSmall( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, SReal fact=1.0 );

    ////////////// large displacements method
    void initLarge(int i, Index&a, Index&b, Index&c, Index&d);
    void computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c);
    void accumulateForceLarge( Vector& f, const Vector & p, Index elementIndex );
    void applyStiffnessLarge( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, SReal fact=1.0 );

    ////////////// polar decomposition method
    void initPolar(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForcePolar( Vector& f, const Vector & p,Index elementIndex );
    void applyStiffnessPolar( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, SReal fact=1.0 );

    void printStiffnessMatrix(int idTetra);

};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TetrahedralCorotationalFEMForceField<sofa::defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_CPP)


} // namespace sofa::component::solidmechanics::fem::elastic
