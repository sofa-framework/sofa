/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>

#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TopologyData.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

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
* indices ordering (same as in HexahedronSetTopology):
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
class HexahedralFEMForceField : virtual public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HexahedralFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef helper::ReadAccessor< DataVecCoord > RDataRefVecCoord;
    typedef helper::WriteAccessor< DataVecDeriv > WDataRefVecDeriv;

    typedef core::topology::BaseMeshTopology::index_type Index;
    typedef core::topology::BaseMeshTopology::Hexa Element;
    typedef core::topology::BaseMeshTopology::SeqHexahedra VecElement;

    typedef defaulttype::Vec<24, Real> Displacement;		///< the displacement vector

    typedef defaulttype::Mat<6, 6, Real> MaterialStiffness;	///< the matrix of material stiffness
    typedef helper::vector<MaterialStiffness> VecMaterialStiffness;  ///< a vector of material stiffness matrices
    typedef defaulttype::Mat<24, 24, Real> ElementMass;

    typedef defaulttype::Mat<24, 24, Real> ElementStiffness;
    typedef helper::vector<ElementStiffness> VecElementStiffness;


    enum
    {
        LARGE = 0,   ///< Symbol of large displacements hexahedron solver
        POLAR = 1,   ///< Symbol of polar displacements hexahedron solver
    };

protected:




    typedef defaulttype::Mat<3, 3, Real> Mat33;
    typedef Mat33 Transformation; ///< matrix for rigid transformations like rotations

    typedef std::pair<int,Real> Col_Value;
    typedef helper::vector< Col_Value > CompressedValue;
    typedef helper::vector< CompressedValue > CompressedMatrix;

    /// the information stored for each hexahedron
    class HexahedronInformation
    {
    public:
        /// material stiffness matrices of each hexahedron
        MaterialStiffness materialMatrix;

        // large displacement method
        helper::fixed_array<Coord,8> rotatedInitialElements;

        Transformation rotation;
        ElementStiffness stiffness;

        HexahedronInformation() {}

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const HexahedronInformation& /*hi*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, HexahedronInformation& /*hi*/ )
        {
            return in;
        }
    };


    HexahedralFEMForceField();
    virtual ~HexahedralFEMForceField();
public:
    void setPoissonRatio(Real val) { this->f_poissonRatio.setValue(val); }

    void setYoungModulus(Real val) { this->f_youngModulus.setValue(val); }

    void setMethod(int val) { method = val; }

    virtual void init();
    virtual void reinit();

    virtual void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    virtual void addDForce (const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx);

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    void draw(const core::visual::VisualParams* vparams);

protected:

    virtual void computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const defaulttype::Vec<8,Coord> &nodes);
    Mat33 integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  );

    /// compute the hookean material matrix
    void computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio);

    void computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K );


    ////////////// large displacements method
    void initLarge(const int i);
    void computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey);
    virtual void accumulateForceLarge( WDataRefVecDeriv& f, RDataRefVecCoord& p, const int i);

    ////////////// polar decomposition method
    void initPolar(const int i);
    void computeRotationPolar( Transformation &r, defaulttype::Vec<8,Coord> &nodes);
    virtual void accumulateForcePolar( WDataRefVecDeriv& f, RDataRefVecCoord & p, const int i);

public:
    int method;
    Data<std::string> f_method; ///< the computation method of the displacements
    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;
    Data<bool> f_drawing;
    /// container that stotes all requires information for each hexahedron
    topology::HexahedronData<sofa::helper::vector<HexahedronInformation> > hexahedronInfo;

    class HFFHexahedronHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Hexahedron,sofa::helper::vector<HexahedronInformation> >
    {
    public:
        typedef typename HexahedralFEMForceField<DataTypes>::HexahedronInformation HexahedronInformation;

        HFFHexahedronHandler(HexahedralFEMForceField<DataTypes>* ff, topology::HexahedronData<sofa::helper::vector<HexahedronInformation> >* data )
            :topology::TopologyDataHandler<core::topology::BaseMeshTopology::Hexahedron,sofa::helper::vector<HexahedronInformation> >(data)
            ,ff(ff)
        {
        }

        void applyCreateFunction(unsigned int, HexahedronInformation &t, const core::topology::BaseMeshTopology::Hexahedron &,
                const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &);
    protected:
        HexahedralFEMForceField<DataTypes>* ff;
    };



protected:
    HFFHexahedronHandler* hexahedronHandler;

    topology::HexahedronSetTopologyContainer* _topology;

    defaulttype::Mat<8,3,int> _coef; ///< coef of each vertices to compute the strain stress matrix
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_SIMPLE_FEM_API HexahedralFEMForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_SIMPLE_FEM_API HexahedralFEMForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_H
