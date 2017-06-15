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
#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/map.h>

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




/** Compute Finite Element forces based on tetrahedral elements.
 */
template<class DataTypes>
class TetrahedralCorotationalFEMForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TetrahedralCorotationalFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    enum { SMALL = 0, ///< Symbol of small displacements tetrahedron solver
            LARGE = 1, ///< Symbol of large displacements tetrahedron solver
            POLAR = 2  ///< Symbol of polar displacements tetrahedron solver
         };
protected:

    /// @name Per element (tetrahedron) data
    /// @{

    /// Displacement vector (deformation of the 4 corners of a tetrahedron
    typedef defaulttype::VecNoInit<12, Real> Displacement;

    /// Material stiffness matrix of a tetrahedron
    typedef defaulttype::Mat<6, 6, Real> MaterialStiffness;

    /// Strain-displacement matrix
    typedef defaulttype::Mat<12, 6, Real> StrainDisplacementTransposed;

    /// Rigid transformation (rotation) matrix
    typedef defaulttype::MatNoInit<3, 3, Real> Transformation;

    /// Stiffness matrix ( = RJKJtRt  with K the Material stiffness matrix, J the strain-displacement matrix, and R the transformation matrix if any )
    typedef defaulttype::Mat<12, 12, Real> StiffnessMatrix;

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
        helper::fixed_array<Coord,4> rotatedInitialElements;
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
    /// container that stotes all requires information for each tetrahedron
    topology::TetrahedronData<sofa::helper::vector<TetrahedronInformation> > tetrahedronInfo;

    /// @name Full system matrix assembly support
    /// @{

    typedef std::pair<int,Real> Col_Value;
    typedef helper::vector< Col_Value > CompressedValue;
    typedef helper::vector< CompressedValue > CompressedMatrix;
    typedef unsigned int Index;

    CompressedMatrix _stiffnesses;
    /// @}

    SReal m_potentialEnergy;

    sofa::core::topology::BaseMeshTopology* _topology;
public:
    class TetrahedronHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Tetrahedron, sofa::helper::vector<TetrahedronInformation> >
    {
    public :
        typedef typename TetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation TetrahedronInformation;
        TetrahedronHandler(TetrahedralCorotationalFEMForceField<DataTypes>* ff,
                           topology::TetrahedronData<sofa::helper::vector<TetrahedronInformation> >* data)
            :topology::TopologyDataHandler<core::topology::BaseMeshTopology::Tetrahedron, sofa::helper::vector<TetrahedronInformation> >(data)
            ,ff(ff)
        {

        }

        void applyCreateFunction(unsigned int, TetrahedronInformation &t, const core::topology::BaseMeshTopology::Tetrahedron &,
                const sofa::helper::vector<unsigned int> &,
                const sofa::helper::vector<double> &);

    protected:
        TetrahedralCorotationalFEMForceField<DataTypes>* ff;

    };
public:
    int method;
    Data<std::string> f_method; ///< the computation method of the displacements
    Data<Real> _poissonRatio;
    Data<Real> _youngModulus;
    Data<VecReal> _localStiffnessFactor;
    Data<bool> _updateStiffnessMatrix;
    Data<bool> _assembling;
    Data<bool> f_drawing;
    Data<bool> _displayWholeVolume;
    Data<defaulttype::Vec4f> drawColor1;
    Data<defaulttype::Vec4f> drawColor2;
    Data<defaulttype::Vec4f> drawColor3;
    Data<defaulttype::Vec4f> drawColor4;
    Data<std::map < std::string, sofa::helper::vector<double> > > _volumeGraph;
protected:
    TetrahedralCorotationalFEMForceField();
    TetrahedronHandler* tetrahedronHandler;
public:

    void setPoissonRatio(Real val) { this->_poissonRatio.setValue(val); }

    void setYoungModulus(Real val) { this->_youngModulus.setValue(val); }

    void setMethod(int val) { method = val; }

    void setUpdateStiffnessMatrix(bool val) { this->_updateStiffnessMatrix.setValue(val); }

    void setComputeGlobalMatrix(bool val) { this->_assembling.setValue(val); }

    virtual void init();
    virtual void reinit();

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix *m, SReal kFactor, unsigned int &offset);

    // Getting the rotation of the vertex by averaing the rotation of neighboring elements
    void getRotation(Transformation& R, unsigned int nodeIdx);
    void getRotations() {}
    void getElementRotation(Transformation& R, unsigned int elementIdx);

    // Getting the stiffness matrix of index i
    void getElementStiffnessMatrix(Real* stiffness, unsigned int nodeIdx);
    void getElementStiffnessMatrix(Real* stiffness, core::topology::BaseMeshTopology::Tetrahedron& te);

    void draw(const core::visual::VisualParams* vparams);

protected:

    void computeStrainDisplacement( StrainDisplacementTransposed &J, Coord a, Coord b, Coord c, Coord d );
    Real peudo_determinant_for_coef ( const defaulttype::Mat<2, 3, Real>&  M );

    void computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacementTransposed &J, const Transformation& Rot );

    void computeMaterialStiffness(int i, Index&a, Index&b, Index&c, Index&d);

    /// overloaded by classes with non-uniform stiffness
    virtual void computeMaterialStiffness(MaterialStiffness& materialMatrix, Index&a, Index&b, Index&c, Index&d, SReal localStiffnessFactor=1);

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

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_SIMPLE_FEM_API TetrahedralCorotationalFEMForceField<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_SIMPLE_FEM_API TetrahedralCorotationalFEMForceField<sofa::defaulttype::Vec3fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_H
