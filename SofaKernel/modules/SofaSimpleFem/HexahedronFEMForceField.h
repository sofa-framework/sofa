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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/SparseGridTopology.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/behavior/BaseRotationFinder.h>
#include <sofa/helper/decompose.h>
#include <sofa/core/behavior/RotationMatrix.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
class HexahedronFEMForceField;

template<class DataTypes>
class HexahedronFEMForceFieldInternalData
{
public:
    typedef HexahedronFEMForceField<DataTypes> Main;
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
class HexahedronFEMForceField : virtual public core::behavior::ForceField<DataTypes>, public sofa::core::behavior::BaseRotationFinder
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedronFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

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

    typedef core::topology::BaseMeshTopology::index_type Index;
    typedef core::topology::BaseMeshTopology::Hexa Element;
    typedef core::topology::BaseMeshTopology::SeqHexahedra VecElement;

    typedef defaulttype::Mat<3, 3, Real> Mat33;

    enum
    {
        LARGE = 0,   ///< Symbol of mean large displacements tetrahedron solver (frame = edges mean on the 3 directions)
        POLAR = 1,   ///< Symbol of polar displacements tetrahedron solver
        SMALL = 2,
    };

public:
    typedef Mat33 Transformation; ///< matrix for rigid transformations like rotations

    int method;
    Data<std::string> f_method; ///< the computation method of the displacements
    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;
    Data<bool> f_updateStiffnessMatrix;
    Data<bool> f_assembling;
    Data< sofa::helper::OptionsGroup > _gatherPt; ///< use in GPU version
    Data< sofa::helper::OptionsGroup > _gatherBsize; ///< use in GPU version
    Data<bool> f_drawing; ///<  draw the forcefield if true
    Data<Real> f_drawPercentageOffset; ///< size of the hexa
    bool needUpdateTopology;

public:
    void setPoissonRatio(Real val) { this->f_poissonRatio.setValue(val); }
    void setYoungModulus(Real val) { this->f_youngModulus.setValue(val); }
    void setMethod(int val) ;
    void setUpdateStiffnessMatrix(bool val) { this->f_updateStiffnessMatrix.setValue(val); }
    void setComputeGlobalMatrix(bool val) { this->f_assembling.setValue(val); }

    void init() override;
    void reinit() override;
    void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f,
                           const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce (const core::MechanicalParams* mparams, DataVecDeriv& df,
                            const DataVecDeriv& dx) override;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/,
                                     const DataVecCoord&  /* x */) const override;

    // getPotentialEnergy is implemented for polar method
    SReal getPotentialEnergy(const core::MechanicalParams*) const override;

    const Transformation& getElementRotation(const unsigned elemidx);

    void getNodeRotation(Transformation& R, unsigned int nodeIdx) ;
    void getRotations(defaulttype::BaseMatrix * rotations,int offset = 0) override ;

    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    void draw(const core::visual::VisualParams* vparams) override;

    void handleTopologyChange() override ;

    // Make other overloaded version of getPotentialEnergy() to show up in subclass.
    using InheritForceField::getPotentialEnergy;

protected:
    typedef defaulttype::Vec<24, Real> Displacement;                ///< the displacement vector

    typedef defaulttype::Mat<6, 6, Real> MaterialStiffness;         ///< the matrix of material stiffness
    typedef helper::vector<MaterialStiffness> VecMaterialStiffness; ///< a vector of material stiffness matrices
    VecMaterialStiffness _materialsStiffnesses;                     ///< the material stiffness matrices vector

    typedef defaulttype::Mat<24, 24, Real> ElementStiffness;
    typedef helper::vector<ElementStiffness> VecElementStiffness;
    Data<VecElementStiffness> _elementStiffnesses; ///< Stiffness matrices per element (K_i)

    typedef std::pair<int,Real> Col_Value;
    typedef helper::vector< Col_Value > CompressedValue;
    typedef helper::vector< CompressedValue > CompressedMatrix;
    CompressedMatrix _stiffnesses;
    SReal m_potentialEnergy;

    sofa::core::topology::BaseMeshTopology* _mesh;
    topology::SparseGridTopology* _sparseGrid;
    Data< VecCoord > _initialPoints; ///< the intial positions of the points

    defaulttype::Mat<8,3,int> _coef; ///< coef of each vertices to compute the strain stress matrix

    HexahedronFEMForceFieldInternalData<DataTypes> *data;
    friend class HexahedronFEMForceFieldInternalData<DataTypes>;

protected:
    HexahedronFEMForceField();

    inline const VecElement *getIndexedElements(){ return & (_mesh->getHexahedra()); }

    virtual void computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M,
                                          const helper::fixed_array<Coord,8> &nodes, const int elementIndice,
                                          double stiffnessFactor=1.0);
    Mat33 integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1,
                              const Real u, const Real v, const Real w, const Mat33& J_1  );

    void computeMaterialStiffness(int i);

    void computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K );


    ////////////// large displacements method
    helper::vector<helper::fixed_array<Coord,8> > _rotatedInitialElements;   ///< The initials positions in its frame
    helper::vector<Transformation> _rotations;
    helper::vector<Transformation> _initialrotations;
    void initLarge(int i, const Element&elem);
    void computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey);
    virtual void accumulateForceLarge( WDataRefVecDeriv &f, RDataRefVecCoord &p, int i, const Element&elem  );

    ////////////// polar decomposition method
    void initPolar(int i, const Element&elem);
    void computeRotationPolar( Transformation &r, defaulttype::Vec<8,Coord> &nodes);
    virtual void accumulateForcePolar( WDataRefVecDeriv &f, RDataRefVecCoord &p, int i, const Element&elem  );

    ////////////// small decomposition method
    void initSmall(int i, const Element&elem);
    virtual void accumulateForceSmall( WDataRefVecDeriv &f, RDataRefVecCoord &p, int i, const Element&elem  );

    bool _alreadyInit;
};

#if  !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_CPP)
extern template class SOFA_SIMPLE_FEM_API HexahedronFEMForceField<defaulttype::Vec3Types>;

#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_H
