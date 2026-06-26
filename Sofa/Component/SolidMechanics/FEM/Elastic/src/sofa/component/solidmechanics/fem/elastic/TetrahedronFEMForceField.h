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
#include <sofa/component/solidmechanics/fem/elastic/fwd.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/Mat.h>
#include <sofa/core/behavior/RotationFinder.h>
#include <sofa/helper/OptionsGroup.h>

#include <sofa/helper/ColorMap.h>
#include <sofa/simulation/task/ParallelForEach.h>

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


template<class DataTypes>
class TetrahedronFEMForceField;

/// This class can be overridden if needed for additional storage within template specializations.
template<class DataTypes>
class TetrahedronFEMForceFieldInternalData
{
public:
    typedef TetrahedronFEMForceField<DataTypes> Main;
    void initPtrData(Main * m)
    {
        auto gatherPt = sofa::helper::getWriteOnlyAccessor(m->d_gatherPt);
        auto gatherBsize = sofa::helper::getWriteOnlyAccessor(m->d_gatherBsize);

        gatherPt.wref().setNames({" "});
        gatherBsize.wref().setNames({" "});
    }
};


/** Compute Finite Element forces based on tetrahedral elements.
*   Corotational methods are based on a rotation from world-space to material-space.
*/
template<class DataTypes>
class TetrahedronFEMForceField : public BaseLinearElasticityFEMForceField<DataTypes>, public sofa::core::behavior::RotationFinder<DataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(TetrahedronFEMForceField, DataTypes), SOFA_TEMPLATE(BaseLinearElasticityFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::RotationFinder, DataTypes));

    typedef typename core::behavior::ForceField<DataTypes> InheritForceField;
    using VecCoord = VecCoord_t<DataTypes>;
    using VecDeriv = VecDeriv_t<DataTypes>;
    using VecReal = VecReal_t<DataTypes>;
    using Coord = Coord_t<DataTypes>;
    using Deriv = Deriv_t<DataTypes>;
    using Real = Real_t<DataTypes>;
    using DataVecDeriv = DataVecDeriv_t<DataTypes>;
    using DataVecCoord = DataVecCoord_t<DataTypes>;

    using Vector = VecCoord;

    typedef core::topology::BaseMeshTopology::Tetra Element;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra VecElement;
    typedef core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
    using TetrahedronID = core::topology::BaseMeshTopology::TetrahedronID;

    enum { SMALL = 0,   ///< Symbol of small displacements tetrahedron solver
           LARGE = 1,   ///< Symbol of corotational large displacements tetrahedron solver based on a QR decomposition    -> Nesme et al 2005 "Efficient, Physically Plausible Finite Elements"
           POLAR = 2,   ///< Symbol of corotational large displacements tetrahedron solver based on a polar decomposition -> Muller et al 2004 "Interactive Virtual Materials"
           SVD = 3      ///< Symbol of corotational large displacements tetrahedron solver based on a SVD decomposition   -> inspired from Irving et al 2004 "Invertible Finite Element for Robust Simulation of Large Deformation"
         };

protected:

    /// @name Per element (tetrahedron) data
    /// @{

    /// Displacement vector (deformation of the 4 corners of a tetrahedron
    typedef type::VecNoInit<12, Real> Displacement;

    /// Material stiffness matrix of a tetrahedron
    typedef type::Mat<6, 6, Real> MaterialStiffness;

    /// Strain-displacement matrix
    typedef type::Mat<12, 6, Real> StrainDisplacement;

    type::MatNoInit<3, 3, Real> R0;

    /// Rigid transformation (rotation) matrix
    typedef type::MatNoInit<3, 3, Real> Transformation;

    /// Stiffness matrix ( = RJKJtRt  with K the Material stiffness matrix, J the strain-displacement matrix, and R the transformation matrix if any )
    typedef type::Mat<12, 12, Real> StiffnessMatrix;

    /// Symmetrical tensor written as a vector following the Voigt notation
    typedef type::VecNoInit<6,Real> VoigtTensor;

    /// @}

    /// Vector of material stiffness of each tetrahedron
    typedef type::vector<MaterialStiffness> VecMaterialStiffness;
    typedef type::vector<StrainDisplacement> VecStrainDisplacement;  ///< a vector of strain-displacement matrices

    /// structures used to compute vonMises stress
    typedef type::Mat<4, 4, Real> Mat44;
    typedef type::Mat<3, 3, Real> Mat33;

    /// Vector of material stiffness matrices of each tetrahedron
    VecMaterialStiffness materialsStiffnesses;
    VecStrainDisplacement strainDisplacements;   ///< the strain-displacement matrices vector
    type::vector<Transformation> rotations;

    type::vector<VoigtTensor> _plasticStrains; ///< one plastic strain per element

    /// @name Full system matrix assembly support
    /// @{

    typedef std::pair<Index,Real> Col_Value;
    typedef type::vector< Col_Value > CompressedValue;
    typedef type::vector< CompressedValue > CompressedMatrix;

    CompressedMatrix _stiffnesses;
    /// @}

    SReal m_potentialEnergy;

    const VecElement *_indexedElements;
    bool needUpdateTopology;

    TetrahedronFEMForceFieldInternalData<DataTypes> data;
    friend class TetrahedronFEMForceFieldInternalData<DataTypes>;

    Real m_restVolume;
    sofa::helper::ColorMap* m_VonMisesColorMap;

    Transformation InvalidTransform;
    type::fixed_array<Coord, 4> InvalidCoords;
    MaterialStiffness InvalidMaterialStiffness;
    StrainDisplacement InvalidStrainDisplacement;

    std::vector< sofa::type::Vec3 > m_renderedPoints;
    std::vector< sofa::type::RGBAColor > m_renderedColors;

public:
    // get the volume of the mesh
    Real getRestVolume() {return m_restVolume;}

    //For a faster contact handling with simplified compliance
    void getRotation(Mat33& R, Index nodeIdx);
    void getRotations(VecReal& vecR) ;
    
    // BaseRotationFinder API
    void getRotations(linearalgebra::BaseMatrix * rotations,int offset = 0) override;
    // RotationFinder<T> API
    type::vector< Mat33 > m_rotations;
    const type::vector<Mat33>& getRotations() override;

    Data< VecCoord > d_initialPoints; ///< Initial Position
    int method;

    Data<std::string> d_method; ///< "small", "large" (by QR), "polar" or "svd" displacements

    Data<VecReal> d_localStiffnessFactor; ///< Allow specification of different stiffness per element. If there are N element and M values are specified, the youngModulus factor for element i would be localStiffnessFactor[i*M/N]
    Data<bool> d_updateStiffnessMatrix;
    Data<bool> d_assembling;

    /// @name Plasticity such as "Interactive Virtual Materials", Muller & Gross, GI 2004
    /// @{
    Data<Real> d_plasticMaxThreshold;
    Data<Real> d_plasticYieldThreshold; ///< Plastic Yield Threshold (2-norm of the strain)
    Data<Real> d_plasticCreep; ///< Plastic Creep Factor * dt [0,1]. Warning this factor depends on dt.
    /// @}

    Data< sofa::helper::OptionsGroup > d_gatherPt; ///< number of dof accumulated per threads during the gather operation (Only use in GPU version)
    Data< sofa::helper::OptionsGroup > d_gatherBsize; ///< number of dof accumulated per threads during the gather operation (Only use in GPU version)
    Data< bool > d_drawHeterogeneousTetra; ///< Draw Heterogeneous Tetra in different color

    Real minYoung, maxYoung;

    /// to compute vonMises stress for visualization
    /// two options: either using corotational strain (TODO)
    ///              or full Green strain tensor (which must be therefore computed for each element and requires some pre-calculations in reinit)
    type::vector<Real> elemLambda;
    type::vector<Real> elemMu;
    type::vector<Mat44> elemShapeFun;

    Real prevMaxStress;

    Data<int> d_computeVonMisesStress; ///< compute and display von Mises stress: 0: no computations, 1: using corotational strain, 2: using full Green strain. Set listening=1
    Data<type::vector<Real> > d_vonMisesPerElement; ///< von Mises Stress per element
    Data<type::vector<Real> > d_vonMisesPerNode; ///< von Mises Stress per node
    Data<type::vector<type::RGBAColor> > d_vonMisesStressColors; ///< Vector of colors describing the VonMises stress

    Real m_minVonMisesPerNode;
    Real m_maxVonMisesPerNode;

    Data<std::string> d_showStressColorMap; ///< Color map used to show stress values
    Data<float> d_showStressAlpha; ///< Alpha for vonMises visualisation
    Data<bool> d_showVonMisesStressPerNode; ///< draw points showing vonMises stress interpolated in nodes
    Data<bool> d_showVonMisesStressPerNodeColorMap; ///< draw elements showing vonMises stress interpolated in nodes
    Data<bool> d_showVonMisesStressPerElement; ///< draw triangles showing vonMises stress interpolated in elements

    Data<Real> d_showElementGapScale; ///< draw gap between elements (when showWireFrame is disabled) [0,1]: 0: no gap, 1: no element

    Data<bool>  d_updateStiffness; ///< update structures (precomputed in init) using stiffness parameters in each iteration (set listening=1)

    using Inherit1::l_topology;

    type::vector<type::Vec<6,Real> > elemDisplacements;

    bool updateVonMisesStress;

protected:
    TetrahedronFEMForceField() ;
    ~TetrahedronFEMForceField() override;

public:
    void setComputeGlobalMatrix(bool val) { this->d_assembling.setValue(val); }

    //for tetra mapping, should be removed in future
    const Transformation& getActualTetraRotation(Index index);
    const Transformation& getInitialTetraRotation(Index index);

    const MaterialStiffness& getMaterialStiffness(TetrahedronID tetraId);
    const StrainDisplacement& getStrainDisplacement(TetrahedronID tetraId);

    // large displacements method
    const type::fixed_array<Coord, 4>& getRotatedInitialElements(TetrahedronID tetraId);


    void setMethod(std::string methodName);
    void setMethod(int val);

    void setUpdateStiffnessMatrix(bool val) { this->d_updateStiffnessMatrix.setValue(val); }

    void reset() override;
    void init() override;
    void reinit() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;

    // Make other overloaded version of getPotentialEnergy() to show up in subclass.
    using InheritForceField::getPotentialEnergy;
    // getPotentialEnergy is implemented for small method
    SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&   x) const override;

    using Inherit1::addKToMatrix;
    void addKToMatrix(sofa::linearalgebra::BaseMatrix *m, SReal kFactor, unsigned int &offset) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    void draw(const core::visual::VisualParams* vparams) override;

    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;


    // Getting the stiffness matrix of index i
    void getElementStiffnessMatrix(Real* stiffness, Index nodeIdx);

protected:
    void computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d );
    Real peudo_determinant_for_coef ( const type::Mat<2, 3, Real>&  M );

    void computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot );

    virtual void computeMaterialStiffness(Index i, Index&a, Index&b, Index&c, Index&d);


    void computeForce( Displacement &F, const Displacement &Depl, VoigtTensor &plasticStrain, const MaterialStiffness &K, const StrainDisplacement &J );
    void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J, SReal fact );


    ////////////// small displacements method
    void initSmall(Index i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceSmall( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );
    void applyStiffnessSmall( Vector& f, const Vector& x, Index i=0, Index a=0,Index b=1,Index c=2,Index d=3, SReal fact=1.0  );

    ////////////// large displacements method
    type::vector<type::fixed_array<Coord,4> > _rotatedInitialElements;   ///< The initials positions in its frame
    type::vector<Transformation> _initialRotations;
    void initLarge(Index i, Index&a, Index&b, Index&c, Index&d);
    void computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c);
    void accumulateForceLarge( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );

    ////////////// polar decomposition method
    type::vector<unsigned int> _rotationIdx;
    void initPolar(Index i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForcePolar( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );

    ////////////// svd decomposition method
    type::vector<Transformation>  _initialTransformation;
    void initSVD(Index i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceSVD( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );

    void applyStiffnessCorotational( Vector& f, const Vector& x, Index i=0, Index a=0,Index b=1,Index c=2,Index d=3, SReal fact=1.0  );

    void handleTopologyChange() override { needUpdateTopology = true; }

    void computeVonMisesStress();
    bool isComputeVonMisesStressMethodSet();
    void computeMinMaxFromYoungsModulus();
    virtual void drawTrianglesFromTetrahedra(const core::visual::VisualParams* vparams,
                                     bool showVonMisesStressPerElement,
                                     bool drawVonMisesStress, const VecCoord& x,
                                     const VecReal& youngModulus,
                                     bool heterogeneous, Real minVM, Real maxVM,
                                     helper::ReadAccessor<Data<type::vector<Real>>> vM);
    virtual void drawTrianglesFromRangeOfTetrahedra(const simulation::Range<VecElement::const_iterator>& range,
                                 const core::visual::VisualParams* vparams,
                                 bool showVonMisesStressPerElement,
                                 bool drawVonMisesStress, bool showWireFrame, const VecCoord& x,
                                 const VecReal& youngModulus,
                                 bool heterogeneous, Real minVM, Real maxVM,
                                 helper::ReadAccessor<Data<type::vector<Real>>> vM);
    void handleEvent(core::objectmodel::Event *event) override;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TetrahedronFEMForceField<defaulttype::Vec3Types>;
#endif

} //namespace sofa::component::solidmechanics::fem::elastic
