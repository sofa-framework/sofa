/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_HEXAHEDRALFEMFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/component/topology/SparseGridTopology.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

#include <sofa/component/topology/HexahedronData.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using sofa::helper::vector;

using namespace sofa::component::topology;

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
class HexahedralFEMForceField : virtual public core::componentmodel::behavior::ForceField<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::componentmodel::topology::BaseMeshTopology::index_type Index;
#ifdef SOFA_NEW_HEXA
    typedef core::componentmodel::topology::BaseMeshTopology::Hexa Element;
    typedef core::componentmodel::topology::BaseMeshTopology::SeqHexas VecElement;
#else
    typedef core::componentmodel::topology::BaseMeshTopology::Cube Element;
    typedef core::componentmodel::topology::BaseMeshTopology::SeqCubes VecElement;
#endif
    static const int LARGE = 0;   ///< Symbol of large displacements hexahedron solver
    static const int POLAR = 1;   ///< Symbol of polar displacements hexahedron solver

protected:
    //component::MechanicalObject<DataTypes>* object;

    typedef Vec<24, Real> Displacement;		///< the displacement vector

    typedef Mat<6, 6, Real> MaterialStiffness;	///< the matrix of material stiffness
    typedef vector<MaterialStiffness> VecMaterialStiffness;         ///< a vector of material stiffness matrices
    //VecMaterialStiffness _materialsStiffnesses;					///< the material stiffness matrices vector

    typedef Mat<24, 24, Real> ElementStiffness;
    typedef vector<ElementStiffness> VecElementStiffness;
    //VecElementStiffness _elementStiffnesses;

    typedef Mat<3, 3, Real> Mat33;
    typedef Mat33 Transformation; ///< matrix for rigid transformations like rotations

    /// the information stored for each hexahedron
    class HexahedronInformation
    {
    public:
        /// material stiffness matrices of each hexahedron
        MaterialStiffness materialMatrix;
        ///< the strain-displacement matrices vector
        //StrainDisplacement strainDisplacementMatrix;
        // large displacement method
        helper::fixed_array<Coord,8> rotatedInitialElements;

        Transformation rotation;
        ElementStiffness stiffness;
        /// polar method
        //Transformation initialTransformation;

        HexahedronInformation()
        {
        }
    };
    /// container that stotes all requires information for each hexahedron
    HexahedronData<HexahedronInformation> hexahedronInfo;


    typedef std::pair<int,Real> Col_Value;
    typedef vector< Col_Value > CompressedValue;
    typedef vector< CompressedValue > CompressedMatrix;
    //CompressedMatrix _stiffnesses;
    double m_potentialEnergy;

    sofa::core::componentmodel::topology::BaseMeshTopology* _topology;

    //topology::SparseGridTopology* _sparseGrid;

    //const VecElement *_indexedElements;
    //Data< VecCoord > _initialPoints; ///< the intial positions of the points


    Mat<8,3,int> _coef; ///< coef of each vertices to compute the strain stress matrix

public:

    int method;
    Data<std::string> f_method; ///< the computation method of the displacements
    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;
//         Data<bool> f_updateStiffnessMatrix;
    Data<bool> f_assembling;


    HexahedralFEMForceField()
        : f_method(initData(&f_method,std::string("large"),"method","\"large\" or \"polar\" displacements"))
        , f_poissonRatio(initData(&f_poissonRatio,(Real)0.45f,"poissonRatio",""))
        , f_youngModulus(initData(&f_youngModulus,(Real)5000,"youngModulus",""))
//             , f_updateStiffnessMatrix(initData(&f_updateStiffnessMatrix,false,"updateStiffnessMatrix",""))
        , f_assembling(initData(&f_assembling,false,"assembling",""))
    {
        _coef[0][0]=-1;
        _coef[1][0]=1;
        _coef[2][0]=1;
        _coef[3][0]=-1;
        _coef[4][0]=-1;
        _coef[5][0]=1;
        _coef[6][0]=1;
        _coef[7][0]=-1;
        _coef[0][1]=-1;
        _coef[1][1]=-1;
        _coef[2][1]=1;
        _coef[3][1]=1;
        _coef[4][1]=-1;
        _coef[5][1]=-1;
        _coef[6][1]=1;
        _coef[7][1]=1;
        _coef[0][2]=-1;
        _coef[1][2]=-1;
        _coef[2][2]=-1;
        _coef[3][2]=-1;
        _coef[4][2]=1;
        _coef[5][2]=1;
        _coef[6][2]=1;
        _coef[7][2]=1;

        _alreadyInit=false;
    }

    void parse(core::objectmodel::BaseObjectDescription* arg);

    void setPoissonRatio(Real val) { this->f_poissonRatio.setValue(val); }

    void setYoungModulus(Real val) { this->f_youngModulus.setValue(val); }

    void setMethod(int val) { method = val; }

// 		void setUpdateStiffnessMatrix(bool val) { this->f_updateStiffnessMatrix.setValue(val); }

    void setComputeGlobalMatrix(bool val) { this->f_assembling.setValue(val); }

    //	component::MechanicalObject<DataTypes>* getObject() { return object; }

    virtual void init();
    virtual void reinit();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    // handle topological changes
    virtual void handleTopologyChange();

    void draw();

protected:


    virtual void computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const Vec<8,Coord> &nodes, const int elementIndice);
    Mat33 integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  );

    void computeMaterialStiffness(int i);

    void computeForce( Displacement &F, const Displacement &Depl, const ElementStiffness &K );


    ////////////// large displacements method
    //vector<helper::fixed_array<Coord,8> > _rotatedInitialElements;   ///< The initials positions in its frame
    //vector<Transformation> _rotations;
    void initLarge(int i);
    void computeRotationLarge( Transformation &r, Coord &edgex, Coord &edgey);
    virtual void accumulateForceLarge( Vector& f, const Vector & p, int i);

    ////////////// polar decomposition method
    void initPolar(int i);
    void computeRotationPolar( Transformation &r, Vec<8,Coord> &nodes);
    virtual void accumulateForcePolar( Vector& f, const Vector & p, int i);

    bool _alreadyInit;

    /// the callback function called when a hexahedron is created
    static void FHexahedronCreationFunction (int , void* ,
            HexahedronInformation &,
            const Hexahedron& , const helper::vector< unsigned int > &, const helper::vector< double >&);

};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
