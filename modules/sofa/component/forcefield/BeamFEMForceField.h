/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H

#include <sofa/component/topology/EdgeData.inl>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/container/StiffnessContainer.h>
#include <sofa/component/container/PoissonContainer.h>
#include <sofa/component/container/LengthContainer.h>
#include <sofa/component/container/RadiusContainer.h>
#include <sofa/core/objectmodel/Data.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using sofa::helper::vector;

/** Compute Finite Element forces based on 6D beam elements.
*/
template<class DataTypes>
class BeamFEMForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BeamFEMForceField,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;
    typedef Data<VecCoord>                  DataVecCoord;
    typedef Data<VecDeriv>                  DataVecDeriv;
    typedef VecCoord Vector;


    typedef unsigned int Index;
    typedef topology::Edge Element;
    typedef sofa::helper::vector<topology::Edge> VecElement;
    typedef helper::vector<unsigned int> VecIndex;
    typedef Vec<3, Real> Vec3;

protected:

    typedef Vec<12, Real> Displacement;        ///< the displacement vector

    //typedef Mat<6, 6, Real> MaterialStiffness;    ///< the matrix of material stiffness
    //typedef vector<MaterialStiffness> VecMaterialStiffness;         ///< a vector of material stiffness matrices
    //VecMaterialStiffness _materialsStiffnesses;                    ///< the material stiffness matrices vector

    //typedef Mat<12, 6, Real> StrainDisplacement;    ///< the strain-displacement matrix
    //typedef vector<StrainDisplacement> VecStrainDisplacement;        ///< a vector of strain-displacement matrices
    //VecStrainDisplacement _strainDisplacements;                       ///< the strain-displacement matrices vector

    typedef Mat<3, 3, Real> Transformation; ///< matrix for rigid transformations like rotations


    typedef Mat<12, 12, Real> StiffnessMatrix;
    //typedef topology::EdgeData<StiffnessMatrix> VecStiffnessMatrices;         ///< a vector of stiffness matrices
    //VecStiffnessMatrices _stiffnessMatrices;                    ///< the material stiffness matrices vector

    struct BeamInfo
    {
        // 	static const double FLEXIBILITY=1.00000; // was 1.00001
        double _E0,_E; //Young
        double _nu;//Poisson
        double _L; //length
        double _r; //radius of the section
        double _rInner; //inner radius of the section if beam is hollow
        double _G; //shear modulus
        double _Iy;
        double _Iz; //Iz is the cross-section moment of inertia (assuming mass ratio = 1) about the z axis;
        double _J;  //Polar moment of inertia (J = Iy + Iz)
        double _A; // A is the cross-sectional area;
        double _Asy; //_Asy is the y-direction effective shear area =  10/9 (for solid circular section) or 0 for a non-Timoshenko beam
        double _Asz; //_Asz is the z-direction effective shear area;
        StiffnessMatrix _k_loc;
        //new: k_loc is the stiffness in the local frame... to compute Ke we only change lambda
        //NewMAT::Matrix  _k_loc;

        // _eigenvalue_loc are 4 diagonal matrices (6x6) representing the eigenvalues of each
        // 6x6 block of _k_loc. _eigenvalue_loc[1] = _eigenvalue_loc[2] since _k_loc[1] = _k_loc[2]
        //NewMAT::DiagonalMatrix  _eigenvalue_loc[4], _inv_eigenvalue_loc[4];
        // k_flex is the stiffness matrix + reinforcement of diagonal (used in gauss-seidel process)
        //NewMAT::Matrix  _k_flex;
        //lambda is a matrix that contains the direction of the local frame in the global frame
        //NewMAT::Matrix  _lambda;
        //non-linear value of the internal forces (computed with previous time step positions) (based on k_loc)
        //NewMAT::ColumnVector  _f_k;
        //initial deformation of the beam (gives the curvature) on the local frame
        //NewMAT::ColumnVector _u_init;
        //actual deformation of the beam on the local frame
        //NewMAT::ColumnVector _u_actual;

        //NewMAT::Matrix _Ke;

        Quat quat;

        //void localStiffness();
        void init(double E, double L, double nu, double r, double rInner);
        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const BeamInfo& /*bi*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, BeamInfo& /*bi*/ )
        {
            return in;
        }
    };

    //just for draw forces
    VecDeriv _forces;

    topology::EdgeData<BeamInfo> beamsData;

    const VecElement *_indexedElements;
    unsigned int maxPoints;
    int _method; ///< the computation method of the displacements
    Data<Real> _poissonRatio;
    Data<Real> _youngModulus;
    Data<bool> _timoshenko;
    Data<Real> _radius;
    Data<Real> _radiusInner;
    Data< VecIndex > _list_segment;
    bool _partial_list_segment;

    bool _updateStiffnessMatrix;
    bool _assembling;

    container::StiffnessContainer* stiffnessContainer;
    container::LengthContainer* lengthContainer;
    container::PoissonContainer* poissonContainer;
    container::RadiusContainer* radiusContainer;

    Quat& beamQuat(int i)
    {
        helper::vector<BeamInfo>& bd = *(beamsData.beginEdit());
        return bd[i].quat;
    }
    sofa::core::topology::BaseMeshTopology* _topology;

public:
    BeamFEMForceField()
        : _indexedElements(NULL)
        , _method(0)
        , _poissonRatio(initData(&_poissonRatio,(Real)0.49f,"poissonRatio","Potion Ratio"))
        , _youngModulus(initData(&_youngModulus,(Real)5000,"youngModulus","Young Modulus"))
        , _timoshenko(initData(&_timoshenko,true,"timoshenko","use Timoshenko beam (non-null section shear area)"))
        , _radius(initData(&_radius,(Real)0.1,"radius","radius of the section"))
        , _radiusInner(initData(&_radiusInner,(Real)0.0,"radiusInner","inner radius of the section for hollow beams"))
        , _list_segment(initData(&_list_segment,"listSegment", "apply the forcefield to a subset list of beam segments. If no segment defined, forcefield applies to the whole topology"))
        , _partial_list_segment(false)
        , _updateStiffnessMatrix(true)
        , _assembling(false)
    {

    }

    BeamFEMForceField(Real poissonRatio, Real youngModulus, Real radius, Real radiusInner)
        : _indexedElements(NULL)
        , _method(0)
        , _poissonRatio(initData(&_poissonRatio,(Real)poissonRatio,"poissonRatio","Potion Ratio"))
        , _youngModulus(initData(&_youngModulus,(Real)youngModulus,"youngModulus","Young Modulus"))
        , _timoshenko(initData(&_timoshenko,true,"timoshenko","use Timoshenko beam (non-null section shear area)"))
        , _radius(initData(&_radius,(Real)radius,"radius","radius of the section"))
        , _radiusInner(initData(&_radiusInner,(Real)radiusInner,"radiusInner","inner radius of the section for hollow beams"))
        , _list_segment(initData(&_list_segment,"listSegment", "apply the forcefield to a subset list of beam segments. If no segment defined, forcefield applies to the whole topology"))
        , _partial_list_segment(false)
        , _updateStiffnessMatrix(true)
        , _assembling(false)
    {
    }

    void setUpdateStiffnessMatrix(bool val) { this->_updateStiffnessMatrix = val; }

    void setComputeGlobalMatrix(bool val) { this->_assembling= val; }

    virtual void init();
    virtual void reinit();
    virtual void reinitBeam(unsigned int i);
    virtual void handleTopologyChange();

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV );
    virtual void addDForce(const sofa::core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv&   datadF , const DataVecDeriv&   datadX );
    virtual void addKToMatrix(const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix );

    void draw(const core::visual::VisualParams* vparams);

    void setBeam(unsigned int i, double E, double L, double nu, double r, double rInner);
    void initBeams(unsigned int size);

protected:

    void drawElement(int i, std::vector< Vector3 >* points, const VecCoord& x);

    //void computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c, Coord d );
    Real peudo_determinant_for_coef ( const Mat<2, 3, Real>&  M );

    //void computeStiffnessMatrix( StiffnessMatrix& S,StiffnessMatrix& SR,const MaterialStiffness &K, const StrainDisplacement &J, const Transformation& Rot );

    //void computeMaterialStiffness(int i, Index&a, Index&b);
    void computeStiffness(int i, Index a, Index b);

    //void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J );

    ////////////// large displacements method
    //vector<fixed_array<Coord,4> > _rotatedInitialElements;   ///< The initials positions in its frame
    //VecReal _initialLength;
    vector<Transformation> _nodeRotations;
    //vector<Quat> _beamQuat;
    void initLarge(int i, Index a, Index b);
    //void computeRotationLarge( Transformation &r, const Vector &p, Index a, Index b);
    void accumulateForceLarge( VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    //void accumulateDampingLarge( Vector& f, Index elementIndex );
    void applyStiffnessLarge( VecDeriv& f, const VecDeriv& x, int i, Index a, Index b, double fact=1.0);

    //sofa::helper::vector< sofa::helper::vector <Real> > subMatrix(unsigned int fr, unsigned int lr, unsigned int fc, unsigned int lc);

    static void BeamFEMEdgeCreationFunction(int edgeIndex, void* param, BeamInfo &ei,
            const topology::Edge& ,  const sofa::helper::vector< unsigned int > &,
            const sofa::helper::vector< double >&);

};

#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_FORCEFIELD_API BeamFEMForceField<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_FORCEFIELD_API BeamFEMForceField<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H
