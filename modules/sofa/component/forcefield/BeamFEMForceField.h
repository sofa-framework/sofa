#ifndef SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_BEAMFEMFORCEFIELD_H

#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/component/topology/EdgeData.inl>
#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/component/topology/FittedRegularGridTopology.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/StiffnessContainer.h>
#include <sofa/component/PoissonContainer.h>
#include <sofa/component/LengthContainer.h>
#include <sofa/component/RadiusContainer.h>
#include "NewMAT/newmat.h"



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
class BeamFEMForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename sofa::defaulttype::Vector3::value_type Real_Sofa;

    typedef unsigned int Index;
    typedef topology::Edge Element;
    typedef sofa::helper::vector<topology::Edge> VecElement;


protected:
    //component::MechanicalObject<DataTypes>* object;

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

        //void localStiffness();
        void init(double E, double L, double nu, double r);
    };

    //just for draw forces
    VecDeriv _forces;

    topology::EdgeData<BeamInfo> beamsData;

    const VecElement *_indexedElements;
    Data< VecCoord > _initialPoints; ///< the intial positions of the points
    int _method; ///< the computation method of the displacements
    Data<Real> _poissonRatio;
    Data<Real> _youngModulus;
    Data<bool> _timoshenko;
    Data<Real> _radius;
    bool _updateStiffnessMatrix;
    bool _assembling;

    StiffnessContainer* stiffnessContainer;
    LengthContainer* lengthContainer;
    PoissonContainer* poissonContainer;
    RadiusContainer* radiusContainer;

public:
    BeamFEMForceField()
        : _indexedElements(NULL)
        , _initialPoints(initData(&_initialPoints, "initialPoints", "Initial Position"))
        , _method(0)
        , _poissonRatio(initData(&_poissonRatio,(Real)0.49f,"poissonRatio","Potion Ratio"))
        , _youngModulus(initData(&_youngModulus,(Real)5000,"youngModulus","Young Modulus"))
        , _timoshenko(initData(&_timoshenko,true,"timoshenko","use Timoshenko beam (non-null section shear area)"))
        , _radius(initData(&_radius,(Real)0.1,"radius","radius of the section"))
        , _updateStiffnessMatrix(true)
        , _assembling(false)
    {

    }

    void setUpdateStiffnessMatrix(bool val) { this->_updateStiffnessMatrix = val; }

    void setComputeGlobalMatrix(bool val) { this->_assembling= val; }

//    component::MechanicalObject<DataTypes>* getObject() { return object; }

    virtual void init();
    virtual void reinit();
    virtual void reinitBeam(unsigned int i);
    virtual void handleTopologyChange();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual sofa::defaulttype::Vector3::value_type getPotentialEnergy(const VecCoord&) { return 0; }

    void addKToMatrix(sofa::defaulttype::BaseMatrix *mat, Real_Sofa k, unsigned int &offset);

    void draw();

    void setBeam(unsigned int i, double E, double L, double nu, double r);
    void initBeams(unsigned int size);

protected:

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
    vector<Quat> _beamQuat;
    void initLarge(int i, Index a, Index b);
    //void computeRotationLarge( Transformation &r, const Vector &p, Index a, Index b);
    void accumulateForceLarge( VecDeriv& f, const VecCoord& x, int i, Index a, Index b);
    //void accumulateDampingLarge( Vector& f, Index elementIndex );
    void applyStiffnessLarge( VecDeriv& f, const VecDeriv& x, int i, Index a, Index b );

    //sofa::helper::vector< sofa::helper::vector <Real> > subMatrix(unsigned int fr, unsigned int lr, unsigned int fc, unsigned int lc);

    static void BeamFEMEdgeCreationFunction(int edgeIndex, void* param, BeamInfo &ei,
            const topology::Edge& ,  const sofa::helper::vector< unsigned int > &,
            const sofa::helper::vector< double >&);

};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
