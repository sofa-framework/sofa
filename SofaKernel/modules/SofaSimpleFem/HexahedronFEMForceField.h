/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifdef SOFA_NEW_HEXA
    typedef core::topology::BaseMeshTopology::Hexa Element;
    typedef core::topology::BaseMeshTopology::SeqHexahedra VecElement;
#else
    typedef core::topology::BaseMeshTopology::Cube Element;
    typedef core::topology::BaseMeshTopology::SeqCubes VecElement;
#endif

    enum
    {
        LARGE = 0,   ///< Symbol of mean large displacements tetrahedron solver (frame = edges mean on the 3 directions)
        POLAR = 1,   ///< Symbol of polar displacements tetrahedron solver
        SMALL = 2,
    };



protected:

    typedef defaulttype::Vec<24, Real> Displacement;		///< the displacement vector

    typedef defaulttype::Mat<6, 6, Real> MaterialStiffness;	///< the matrix of material stiffness
    typedef helper::vector<MaterialStiffness> VecMaterialStiffness;         ///< a vector of material stiffness matrices
    VecMaterialStiffness _materialsStiffnesses;					///< the material stiffness matrices vector

    typedef defaulttype::Mat<24, 24, Real> ElementStiffness;
    typedef helper::vector<ElementStiffness> VecElementStiffness;
    Data<VecElementStiffness> _elementStiffnesses; ///< Stiffness matrices per element (K_i)

    typedef defaulttype::Mat<3, 3, Real> Mat33;


    typedef std::pair<int,Real> Col_Value;
    typedef helper::vector< Col_Value > CompressedValue;
    typedef helper::vector< CompressedValue > CompressedMatrix;
    CompressedMatrix _stiffnesses;
    SReal m_potentialEnergy;


    sofa::core::topology::BaseMeshTopology* _mesh;
    topology::SparseGridTopology* _sparseGrid;
    Data< VecCoord > _initialPoints; ///< the intial positions of the points


    defaulttype::Mat<8,3,int> _coef; ///< coef of each vertices to compute the strain stress matrix
#ifndef SOFA_NEW_HEXA
    static const int _indices[8]; ///< indices ordering is different than in topology node
#endif

    HexahedronFEMForceFieldInternalData<DataTypes> *data;
    friend class HexahedronFEMForceFieldInternalData<DataTypes>;

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

protected:
    HexahedronFEMForceField()
        : _elementStiffnesses(initData(&_elementStiffnesses,"stiffnessMatrices", "Stiffness matrices per element (K_i)"))
        , _mesh(NULL)
        , _sparseGrid(NULL)
        , _initialPoints(initData(&_initialPoints,"initialPoints", "Initial Position"))
        , data(new HexahedronFEMForceFieldInternalData<DataTypes>())
        , f_method(initData(&f_method,std::string("large"),"method","\"large\" or \"polar\" or \"small\" displacements" ))
        , f_poissonRatio(initData(&f_poissonRatio,(Real)0.45f,"poissonRatio",""))
        , f_youngModulus(initData(&f_youngModulus,(Real)5000,"youngModulus",""))
        , f_updateStiffnessMatrix(initData(&f_updateStiffnessMatrix,false,"updateStiffnessMatrix",""))
        , f_assembling(initData(&f_assembling,false,"assembling",""))
        , _gatherPt(initData(&_gatherPt,"gatherPt","number of dof accumulated per threads during the gather operation (Only use in GPU version)"))
        , _gatherBsize(initData(&_gatherBsize,"gatherBsize","number of dof accumulated per threads during the gather operation (Only use in GPU version)"))
        , f_drawing(initData(&f_drawing,true,"drawing"," draw the forcefield if true"))
        , f_drawPercentageOffset(initData(&f_drawPercentageOffset,(Real)0.15,"drawPercentageOffset","size of the hexa"))
        , needUpdateTopology(false)
    {
        data->initPtrData(this);
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

		f_poissonRatio.setRequired(true);
		f_youngModulus.setRequired(true);
    }
public:
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

    virtual void init() override;
    virtual void reinit() override;

    virtual void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    virtual void addDForce (const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    // Make other overloaded version of getPotentialEnergy() to show up in subclass.
    using InheritForceField::getPotentialEnergy;
    // getPotentialEnergy is implemented for polar method
    virtual SReal getPotentialEnergy(const core::MechanicalParams*) const override;

    const Transformation& getElementRotation(const unsigned elemidx);

    void getNodeRotation(Transformation& R, unsigned int nodeIdx)
    {
        core::topology::BaseMeshTopology::HexahedraAroundVertex liste_hexa = _mesh->getHexahedraAroundVertex(nodeIdx);

        R[0][0] = R[1][1] = R[2][2] = 1.0 ;
        R[0][1] = R[0][2] = R[1][0] = R[1][2] = R[2][0] = R[2][1] = 0.0 ;

        unsigned int numHexa=liste_hexa.size();

        for (unsigned int ti=0; ti<numHexa; ti++)
        {
            Transformation R0t;
            R0t.transpose(_initialrotations[liste_hexa[ti]]);
            Transformation Rcur = getElementRotation(liste_hexa[ti]);
            R += Rcur * R0t;
        }

        // on "moyenne"
        R[0][0] = R[0][0]/numHexa ; R[0][1] = R[0][1]/numHexa ; R[0][2] = R[0][2]/numHexa ;
        R[1][0] = R[1][0]/numHexa ; R[1][1] = R[1][1]/numHexa ; R[1][2] = R[1][2]/numHexa ;
        R[2][0] = R[2][0]/numHexa ; R[2][1] = R[2][1]/numHexa ; R[2][2] = R[2][2]/numHexa ;

        defaulttype::Mat<3,3,Real> Rmoy;
        helper::Decompose<Real>::polarDecomposition( R, Rmoy );

        R = Rmoy;
    }

    void getRotations(defaulttype::BaseMatrix * rotations,int offset = 0) override
    {
        unsigned int nbdof = this->mstate->getSize();

        if (component::linearsolver::RotationMatrix<float> * diag = dynamic_cast<component::linearsolver::RotationMatrix<float> *>(rotations))
        {
            Transformation R;
            for (unsigned int e=0; e<nbdof; ++e)
            {
                getNodeRotation(R,e);
                for(int j=0; j<3; j++)
                {
                    for(int i=0; i<3; i++)
                    {
                        diag->getVector()[e*9 + j*3 + i] = (float)R[j][i];
                    }
                }
            }
        }
        else if (component::linearsolver::RotationMatrix<double> * diag = dynamic_cast<component::linearsolver::RotationMatrix<double> *>(rotations))
        {
            Transformation R;
            for (unsigned int e=0; e<nbdof; ++e)
            {
                getNodeRotation(R,e);
                for(int j=0; j<3; j++)
                {
                    for(int i=0; i<3; i++)
                    {
                        diag->getVector()[e*9 + j*3 + i] = R[j][i];
                    }
                }
            }
        }
        else
        {
            for (unsigned int i=0; i<nbdof; ++i)
            {
                Transformation t;
                getNodeRotation(t,i);
                int e = offset+i*3;
                rotations->set(e+0,e+0,t[0][0]); rotations->set(e+0,e+1,t[0][1]); rotations->set(e+0,e+2,t[0][2]);
                rotations->set(e+1,e+0,t[1][0]); rotations->set(e+1,e+1,t[1][1]); rotations->set(e+1,e+2,t[1][2]);
                rotations->set(e+2,e+0,t[2][0]); rotations->set(e+2,e+1,t[2][1]); rotations->set(e+2,e+2,t[2][2]);
            }
        }
    }

    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;


    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

    void draw(const core::visual::VisualParams* vparams) override;

    void handleTopologyChange() override
    {
        needUpdateTopology = true;
    }


protected:


    inline const VecElement *getIndexedElements()
    {
#ifdef SOFA_NEW_HEXA
        return & (_mesh->getHexahedra());
#else
        return & (_mesh->getCubes());
#endif
    }

    virtual void computeElementStiffness( ElementStiffness &K, const MaterialStiffness &M, const helper::fixed_array<Coord,8> &nodes, const int elementIndice, double stiffnessFactor=1.0);
    Mat33 integrateStiffness( int signx0, int signy0, int signz0, int signx1, int signy1, int signz1, const Real u, const Real v, const Real w, const Mat33& J_1  );

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

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_SIMPLE_FEM_API HexahedronFEMForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_SIMPLE_FEM_API HexahedronFEMForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_HEXAHEDRONFEMFORCEFIELD_H
