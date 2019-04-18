/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/behavior/BaseRotationFinder.h>
#include <sofa/core/behavior/RotationMatrix.h>
#include <sofa/helper/OptionsGroup.h>

#include <sofa/helper/ColorMap.h>

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


namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
class TetrahedronFEMForceField;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class TetrahedronFEMForceFieldInternalData
{
public:
    typedef TetrahedronFEMForceField<DataTypes> Main;
    void initPtrData(Main * m)
    {
        m->_gatherPt.beginEdit()->setNames(1," ");
        m->_gatherPt.endEdit();

        m->_gatherBsize.beginEdit()->setNames(1," ");
        m->_gatherBsize.endEdit();
    }
};


/** Compute Finite Element forces based on tetrahedral elements.
*   Corotational methods are based on a rotation from world-space to material-space.
*/
template<class DataTypes>
class TetrahedronFEMForceField : public core::behavior::ForceField<DataTypes>, public sofa::core::behavior::BaseRotationFinder
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(TetrahedronFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes), core::behavior::BaseRotationFinder);

    typedef typename core::behavior::ForceField<DataTypes> InheritForceField;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef core::topology::BaseMeshTopology::index_type Index;
    typedef core::topology::BaseMeshTopology::Tetra Element;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra VecElement;
    typedef core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;

    enum { SMALL = 0,   ///< Symbol of small displacements tetrahedron solver
           LARGE = 1,   ///< Symbol of corotational large displacements tetrahedron solver based on a QR decomposition    -> Nesme et al 2005 "Efficient, Physically Plausible Finite Elements"
           POLAR = 2,   ///< Symbol of corotational large displacements tetrahedron solver based on a polar decomposition -> Muller et al 2004 "Interactive Virtual Materials"
           SVD = 3      ///< Symbol of corotational large displacements tetrahedron solver based on a SVD decomposition   -> inspired from Irving et al 2004 "Invertible Finite Element for Robust Simulation of Large Deformation"
         };

protected:

    /// @name Per element (tetrahedron) data
    /// @{

    /// Displacement vector (deformation of the 4 corners of a tetrahedron
    typedef defaulttype::VecNoInit<12, Real> Displacement;

    /// Material stiffness matrix of a tetrahedron
    typedef defaulttype::Mat<6, 6, Real> MaterialStiffness;

    /// Strain-displacement matrix
    typedef defaulttype::Mat<12, 6, Real> StrainDisplacement;

    defaulttype::MatNoInit<3, 3, Real> R0;

    /// Rigid transformation (rotation) matrix
    typedef defaulttype::MatNoInit<3, 3, Real> Transformation;

    /// Stiffness matrix ( = RJKJtRt  with K the Material stiffness matrix, J the strain-displacement matrix, and R the transformation matrix if any )
    typedef defaulttype::Mat<12, 12, Real> StiffnessMatrix;

    /// Symmetrical tensor written as a vector following the Voigt notation
    typedef defaulttype::VecNoInit<6,Real> VoigtTensor;

    /// @}

    /// Vector of material stiffness of each tetrahedron
    typedef helper::vector<MaterialStiffness> VecMaterialStiffness;
    typedef helper::vector<StrainDisplacement> VecStrainDisplacement;  ///< a vector of strain-displacement matrices

    /// structures used to compute vonMises stress
    typedef defaulttype::Mat<4, 4, Real> Mat44;
    typedef defaulttype::Mat<3, 3, Real> Mat33;
    typedef defaulttype::Mat<4, 3, Real> Mat43;

    /// Vector of material stiffness matrices of each tetrahedron
    VecMaterialStiffness materialsStiffnesses;
    VecStrainDisplacement strainDisplacements;   ///< the strain-displacement matrices vector
    helper::vector<Transformation> rotations;

    helper::vector<VoigtTensor> _plasticStrains; ///< one plastic strain per element

    /// @name Full system matrix assembly support
    /// @{

    typedef std::pair<int,Real> Col_Value;
    typedef helper::vector< Col_Value > CompressedValue;
    typedef helper::vector< CompressedValue > CompressedMatrix;

    CompressedMatrix _stiffnesses;
    /// @}

    SReal m_potentialEnergy;

    core::topology::BaseMeshTopology* _mesh;
    const VecElement *_indexedElements;
    bool needUpdateTopology;

    TetrahedronFEMForceFieldInternalData<DataTypes> data;
    friend class TetrahedronFEMForceFieldInternalData<DataTypes>;

    Real m_restVolume;

public:
    // get the volume of the mesh
    Real getRestVolume() {return m_restVolume;}

    //For a faster contact handling with simplified compliance
    void getRotation(Transformation& R, unsigned int nodeIdx);
    void getRotations(VecReal& vecR) ;
    void getRotations(defaulttype::BaseMatrix * rotations,int offset = 0) override ;
    Data< VecCoord > _initialPoints; ///< the initial positions of the points
    int method;
    Data<std::string> f_method; ///< the computation method of the displacements

    Data<Real> _poissonRatio; ///< FEM Poisson Ratio [0,0.5[
    Data<VecReal > _youngModulus; ///< FEM Young Modulus
    Data<VecReal> _localStiffnessFactor; ///< Allow specification of different stiffness per element. If there are N element and M values are specified, the youngModulus factor for element i would be localStiffnessFactor[i*M/N]
    Data<bool> _updateStiffnessMatrix;
    Data<bool> _assembling;


    /// @name Plasticity such as "Interactive Virtual Materials", Muller & Gross, GI 2004
    /// @{
    Data<Real> _plasticMaxThreshold;
    Data<Real> _plasticYieldThreshold; ///< Plastic Yield Threshold (2-norm of the strain)
    Data<Real> _plasticCreep; ///< this parameters is different from the article, here it includes the multiplication by dt
    /// @}


    Data< sofa::helper::OptionsGroup > _gatherPt; ///< use in GPU version
    Data< sofa::helper::OptionsGroup > _gatherBsize; ///< use in GPU version
    Data< bool > drawHeterogeneousTetra; ///< Draw Heterogeneous Tetra in different color
    Data< bool > drawAsEdges; ///< Draw as edges instead of tetrahedra

    Real minYoung, maxYoung;

    /// to compute vonMises stress for visualization
    /// two options: either using corotational strain (TODO)
    ///              or full Green strain tensor (which must be therefore computed for each element and requires some pre-calculations in reinit)
    helper::vector<Real> elemLambda;
    helper::vector<Real> elemMu;
    helper::vector<Mat44> elemShapeFun;

    Real prevMaxStress;

    Data<int> _computeVonMisesStress; ///< compute and display von Mises stress: 0: no computations, 1: using corotational strain, 2: using full Green strain
    Data<helper::vector<Real> > _vonMisesPerElement; ///< von Mises Stress per element
    Data<helper::vector<Real> > _vonMisesPerNode; ///< von Mises Stress per node
    Data<helper::vector<defaulttype::Vec4f> > _vonMisesStressColors; ///< Vector of colors describing the VonMises stress
    
    helper::ColorMap m_VonMisesColorMap;
    Data<std::string> _showStressColorMap; ///< Color map used to show stress values
    Data<float> _showStressAlpha; ///< Alpha for vonMises visualisation
    Data<bool> _showVonMisesStressPerNode; ///< draw points  showing vonMises stress interpolated in nodes

    /// Suppress field for save as function
    Data<bool>  isToPrint;
    Data<bool>  _updateStiffness; ///< udpate structures (precomputed in init) using stiffness parameters in each iteration (set listening=1)

    SingleLink<TetrahedronFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_topology;

    helper::vector<defaulttype::Vec<6,Real> > elemDisplacements;

    bool updateVonMisesStress;

protected:
    TetrahedronFEMForceField() ;
    ~TetrahedronFEMForceField() override;

public:
    void setPoissonRatio(Real val) { this->_poissonRatio.setValue(val); }
    void setYoungModulus(Real val) ;
    void setComputeGlobalMatrix(bool val) { this->_assembling.setValue(val); }

    //for tetra mapping, should be removed in future
    Transformation getActualTetraRotation(unsigned int index);
    Transformation getInitialTetraRotation(unsigned int index);

    void setMethod(std::string methodName);
    void setMethod(int val);

    void setUpdateStiffnessMatrix(bool val) { this->_updateStiffnessMatrix.setValue(val); }

    void reset() override;
    void init() override;
    void reinit() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;

    // Make other overloaded version of getPotentialEnergy() to show up in subclass.
    using InheritForceField::getPotentialEnergy;
    // getPotentialEnergy is implemented for small method
    SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&   x) const override;

    void addKToMatrix(sofa::defaulttype::BaseMatrix *m, SReal kFactor, unsigned int &offset) override;
    void addKToMatrix(const core::MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/ ) override;

    void addSubKToMatrix(sofa::defaulttype::BaseMatrix *mat, const helper::vector<unsigned> & subMatrixIndex, SReal k, unsigned int &offset) override;


    void draw(const core::visual::VisualParams* vparams) override;

    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;


    // Getting the stiffness matrix of index i
    void getElementStiffnessMatrix(Real* stiffness, unsigned int nodeIdx);
    void getElementStiffnessMatrix(Real* stiffness, Tetrahedron& te);
    virtual void computeMaterialStiffness(MaterialStiffness& materialMatrix, Index&a, Index&b, Index&c, Index&d);

protected:
    void computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d );
    Real peudo_determinant_for_coef ( const defaulttype::Mat<2, 3, Real>&  M );

    void computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot );

    virtual void computeMaterialStiffness(int i, Index&a, Index&b, Index&c, Index&d);


    void computeForce( Displacement &F, const Displacement &Depl, VoigtTensor &plasticStrain, const MaterialStiffness &K, const StrainDisplacement &J );
    void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J, SReal fact );


    ////////////// small displacements method
    void initSmall(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceSmall( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );
    void applyStiffnessSmall( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, SReal fact=1.0  );

    ////////////// large displacements method
    helper::vector<helper::fixed_array<Coord,4> > _rotatedInitialElements;   ///< The initials positions in its frame
    helper::vector<Transformation> _initialRotations;
    void initLarge(int i, Index&a, Index&b, Index&c, Index&d);
    void computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c);
    void accumulateForceLarge( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );

    ////////////// polar decomposition method
    helper::vector<unsigned int> _rotationIdx;
    void initPolar(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForcePolar( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );

    ////////////// svd decomposition method
    helper::vector<Transformation>  _initialTransformation;
    void initSVD(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceSVD( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );

    void applyStiffnessCorotational( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, SReal fact=1.0  );

    void handleTopologyChange() override { needUpdateTopology = true; }

    void computeVonMisesStress();
    void handleEvent(core::objectmodel::Event *event) override;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_CPP)
extern template class SOFA_SIMPLE_FEM_API TetrahedronFEMForceField<defaulttype::Vec3Types>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
