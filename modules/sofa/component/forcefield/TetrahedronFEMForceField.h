/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_H

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/component.h>
#include <sofa/core/behavior/BaseRotationFinder.h>
#include <sofa/core/behavior/RotationMatrix.h>
#include <sofa/helper/OptionsGroup.h>

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

using namespace sofa::defaulttype;
using sofa::helper::vector;
using namespace sofa::core::topology;

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
*/
template<class DataTypes>
class TetrahedronFEMForceField : public core::behavior::ForceField<DataTypes>, public sofa::core::behavior::BaseRotationFinder
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TetrahedronFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

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
    typedef BaseMeshTopology::Tetrahedron Tetrahedron;

    enum { SMALL = 0,   ///< Symbol of small displacements tetrahedron solver
            LARGE = 1,   ///< Symbol of large displacements tetrahedron solver
            POLAR = 2
         }; ///< Symbol of polar displacements tetrahedron solver

protected:

    /// @name Per element (tetrahedron) data
    /// @{

    /// Displacement vector (deformation of the 4 corners of a tetrahedron
    typedef VecNoInit<12, Real> Displacement;

    /// Material stiffness matrix of a tetrahedron
    typedef Mat<6, 6, Real> MaterialStiffness;

    /// Strain-displacement matrix
    typedef Mat<12, 6, Real> StrainDisplacement;

    /// Rigid transformation (rotation) matrix
    typedef MatNoInit<3, 3, Real> Transformation;

    /// Stiffness matrix ( = RJKJtRt  with K the Material stiffness matrix, J the strain-displacement matrix, and R the transformation matrix if any )
    typedef Mat<12, 12, Real> StiffnessMatrix;

    /// @}

    /// Vector of material stiffness of each tetrahedron
    typedef vector<MaterialStiffness> VecMaterialStiffness;
    typedef vector<StrainDisplacement> VecStrainDisplacement;  ///< a vector of strain-displacement matrices

    typedef struct
    {
        /// Vector of material stiffness matrices of each tetrahedron
        VecMaterialStiffness materialsStiffnesses;
        VecStrainDisplacement strainDisplacements;   ///< the strain-displacement matrices vector
        vector<Transformation> rotations;
    } ParallelData;

    ParallelData * parallelDataSimu;
    ParallelData * parallelDataThrd;
    ParallelData * parallelDataInit[2]; //use to remember initial values because parallelData will be erase

    void createParallelData()
    {
        parallelDataInit[1] = new ParallelData();
        parallelDataInit[1]->materialsStiffnesses = parallelDataInit[0]->materialsStiffnesses;
        parallelDataInit[1]->strainDisplacements = parallelDataInit[0]->strainDisplacements;
        parallelDataInit[1]->rotations = parallelDataInit[0]->rotations;
    }

    /// @name Full system matrix assembly support
    /// @{

    typedef std::pair<int,Real> Col_Value;
    typedef vector< Col_Value > CompressedValue;
    typedef vector< CompressedValue > CompressedMatrix;

    CompressedMatrix _stiffnesses;
    /// @}

    SReal m_potentialEnergy;

    core::topology::BaseMeshTopology* _mesh;
    const VecElement *_indexedElements;
    bool needUpdateTopology;

    TetrahedronFEMForceFieldInternalData<DataTypes> data;
    friend class TetrahedronFEMForceFieldInternalData<DataTypes>;

public:
    //For a faster contact handling with simplified compliance
    void getRotation(Transformation& R, unsigned int nodeIdx);

    void getRotations(VecReal& vecR)
    {
        vecR.resize(_indexedElements->size()*9);
        for (unsigned int i=0; i<_indexedElements->size(); ++i)
            getRotation(*(Transformation*)&(vecR[i*9]),i);
    }

    void getRotations(defaulttype::BaseMatrix * rotations,int offset = 0)
    {
        unsigned int nbdof = this->mstate->getX()->size();

        rotations->resize(nbdof*3,nbdof*3);
        if (component::linearsolver::RotationMatrix<Real> * diag = dynamic_cast<component::linearsolver::RotationMatrix<Real> *>(rotations))
        {
            for (unsigned int i=0; i<nbdof; ++i) getRotation(*(Transformation*)&(diag->getVector()[i*9]),i);
        }
        else
        {
            for (unsigned int i=0; i<nbdof; ++i)
            {
                Transformation t;
                getRotation(t,i);
                int e = offset+i*3;
                rotations->set(e+0,e+0,t[0][0]); rotations->set(e+0,e+1,t[0][1]); rotations->set(e+0,e+2,t[0][2]);
                rotations->set(e+1,e+0,t[1][0]); rotations->set(e+1,e+1,t[1][1]); rotations->set(e+1,e+2,t[1][2]);
                rotations->set(e+2,e+0,t[2][0]); rotations->set(e+2,e+1,t[2][1]); rotations->set(e+2,e+2,t[2][2]);
            }
        }
    }

    Data< VecCoord > _initialPoints; ///< the intial positions of the points
    int method;
    Data<std::string> f_method; ///< the computation method of the displacements

    Data<Real> _poissonRatio;
    //Data<Real> _youngModulus;
    Data<VecReal > _youngModulus;
    Data<VecReal> _localStiffnessFactor;
    Data<bool> _updateStiffnessMatrix;
    Data<bool> _assembling;
    Data< sofa::helper::OptionsGroup > _gatherPt; //use in GPU version
    Data< sofa::helper::OptionsGroup > _gatherBsize; //use in GPU version
    Data< bool > drawHeterogeneousTetra;
    Data< bool > drawAsEdges;

    Real minYoung;
    Real maxYoung;
protected:
    TetrahedronFEMForceField()
        : parallelDataSimu(NULL)
        ,parallelDataThrd(NULL)
        ,_mesh(NULL)
        , _indexedElements(NULL)
        , needUpdateTopology(false)
        , _initialPoints(initData(&_initialPoints, "initialPoints", "Initial Position"))
        , f_method(initData(&f_method,std::string("large"),"method","\"small\", \"large\" (by QR) or \"polar\" displacements"))
        , _poissonRatio(initData(&_poissonRatio,(Real)0.45f,"poissonRatio","FEM Poisson Ratio"))
        , _youngModulus(initData(&_youngModulus,"youngModulus","FEM Young Modulus"))
        , _localStiffnessFactor(initData(&_localStiffnessFactor, "localStiffnessFactor","Allow specification of different stiffness per element. If there are N element and M values are specified, the youngModulus factor for element i would be localStiffnessFactor[i*M/N]"))
        , _updateStiffnessMatrix(initData(&_updateStiffnessMatrix,false,"updateStiffnessMatrix",""))
        , _assembling(initData(&_assembling,false,"computeGlobalMatrix",""))
        , _gatherPt(initData(&_gatherPt,"gatherPt","number of dof accumulated per threads during the gather operation (Only use in GPU version)"))
        , _gatherBsize(initData(&_gatherBsize,"gatherBsize","number of dof accumulated per threads during the gather operation (Only use in GPU version)"))
        , drawHeterogeneousTetra(initData(&drawHeterogeneousTetra,false,"drawHeterogeneousTetra","Draw Heterogeneous Tetra in different color"))
        , drawAsEdges(initData(&drawAsEdges,false,"drawAsEdges","Draw as edges instead of tetrahedra"))
    {
        data.initPtrData(this);
        parallelDataInit[0]=0;
        parallelDataInit[1]=0;
        this->addAlias(&_assembling, "assembling");
        minYoung = 0.0;
        maxYoung = 0.0;
    }

    ~TetrahedronFEMForceField()
    {
        if (parallelDataInit[0]) delete parallelDataInit[0];
        if (parallelDataInit[1]) delete parallelDataInit[1];

        parallelDataInit[0] = NULL;
        parallelDataInit[1] = NULL;
        parallelDataThrd = NULL;
        parallelDataInit[2] = NULL;

// 	    if (_gatherPt) delete _gatherPt;
// 	    if (_gatherBsize)  delete _gatherBsize;
// 	    _gatherPt = NULL;
// 	    _gatherBsize = NULL
    }
public:
    void setPoissonRatio(Real val) { this->_poissonRatio.setValue(val); }

    void setYoungModulus(Real val)
    {
        VecReal newY;
        newY.resize(1);
        newY[0] = val;
        _youngModulus.setValue(newY);
    }

    void setComputeGlobalMatrix(bool val) { this->_assembling.setValue(val); }

    //for tetra mapping, should be removed in future
    Transformation getActualTetraRotation(unsigned int index)
    {
        if (index < parallelDataSimu->rotations.size() )
            return parallelDataSimu->rotations[index];
        else { Transformation t; t.identity(); return t; }
    }

    Transformation getInitialTetraRotation(unsigned int index)
    {
        if (index < parallelDataSimu->rotations.size() )
            return _initialRotations[index];
        else { Transformation t; t.identity(); return t; }
    }


    void setMethod(std::string methodName)
    {
        if (methodName == "small")	this->setMethod(SMALL);
        else if (methodName  == "polar")	this->setMethod(POLAR);
        else
        {
            if (methodName != "large")
                serr << "unknown method: large method will be used. Remark: Available method are \"small\", \"polar\", \"large\" "<<sendl;
            this->setMethod(LARGE);
        }
    }

    void setMethod(int val)
    {
        method = val;
        switch(val)
        {
        case SMALL: f_method.setValue("small"); break;
        case POLAR: f_method.setValue("polar"); break;
        default   : f_method.setValue("large");
        };
    }

    void setUpdateStiffnessMatrix(bool val) { this->_updateStiffnessMatrix.setValue(val); }

    virtual void init();
    virtual void reinit();

    virtual void addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix *m, SReal kFactor, unsigned int &offset);

    void draw(const core::visual::VisualParams* vparams);

    // Getting the stiffness matrix of index i
    void getElementStiffnessMatrix(Real* stiffness, unsigned int nodeIdx);
    void getElementStiffnessMatrix(Real* stiffness, Tetrahedron& te);
    virtual void computeMaterialStiffness(MaterialStiffness& materialMatrix, Index&a, Index&b, Index&c, Index&d);

    virtual void handleEvent(sofa::core::objectmodel::Event* event);

protected:

    void computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d );
    Real peudo_determinant_for_coef ( const Mat<2, 3, Real>&  M );

    void computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot );

    virtual void computeMaterialStiffness(int i, Index&a, Index&b, Index&c, Index&d);

    void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J );
    void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J, double fact );

    ////////////// small displacements method
    void initSmall(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceSmall( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );
    void applyStiffnessSmall( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, double fact=1.0  );

    ////////////// large displacements method
    vector<helper::fixed_array<Coord,4> > _rotatedInitialElements;   ///< The initials positions in its frame
    vector<Transformation> _initialRotations;
    void initLarge(int i, Index&a, Index&b, Index&c, Index&d);
    void computeRotationLarge( Transformation &r, const Vector &p, const Index &a, const Index &b, const Index &c);
    void accumulateForceLarge( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );
    void applyStiffnessLarge( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, double fact=1.0 );

    ////////////// polar decomposition method
    vector<Transformation>  _initialTransformation;
    vector<unsigned int> _rotationIdx;
    void initPolar(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForcePolar( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );
    void applyStiffnessPolar( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, double fact=1.0  );

    void handleTopologyChange()
    {
        needUpdateTopology = true;
    }
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_SIMPLE_FEM_API TetrahedronFEMForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_SIMPLE_FEM_API TetrahedronFEMForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
