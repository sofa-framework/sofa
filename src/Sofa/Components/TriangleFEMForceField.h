#pragma once


#include "Sofa/Core/ForceField.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Components/MeshTopology.h"

#include "Common/Vec.h"
#include "Common/Mat.h"

#include "Sofa/Components/Common/SofaBaseMatrix.h"
#include "Sofa/Components/Common/SofaBaseVector.h"


namespace Sofa
{


namespace Components
{


using namespace Common;


template<class DataTypes>
class TriangleFEMForceField : public Core::BasicForceField, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef MeshTopology::index_type Index;
    typedef MeshTopology::Triangle Element;
    typedef MeshTopology::SeqTriangles VecElement;

    static const int SMALL = 1;										///< Symbol of small displacements triangle solver
    static const int LARGE = 0;										///< Symbol of large displacements triangle solver

protected:
    Core::MechanicalObject<DataTypes>* _object;

    typedef Vec<6, Real> Displacement;								///< the displacement vector

    typedef Mat<3, 3, Real> MaterialStiffness;						///< the matrix of material stiffness
    typedef std::vector<MaterialStiffness> VecMaterialStiffness;    ///< a vector of material stiffness matrices
    VecMaterialStiffness _materialsStiffnesses;						///< the material stiffness matrices vector

    typedef Mat<6, 3, Real> StrainDisplacement;						///< the strain-displacement matrix
    typedef std::vector<StrainDisplacement> VecStrainDisplacement;	///< a vector of strain-displacement matrices
    VecStrainDisplacement _strainDisplacements;						///< the strain-displacement matrices vector

    typedef Mat<3, 3, Real > Transformation;						///< matrix for rigid transformations like rotations


    MeshTopology* _mesh;
    const VecElement *_indexedElements;
    VecCoord _initialPoints; ///< the intial positions of the points
//     int _method; ///< the computation method of the displacements
//     Real _poissonRatio;
//     Real _youngModulus;
//     Real _dampingRatio;

public:

    TriangleFEMForceField(Core::MechanicalObject<DataTypes>* object);

    virtual const char* getTypeName() const
    {
        return "TriangleFEMForceField";
    }

    virtual ~TriangleFEMForceField();


    virtual void init();
    virtual void addForce();
    virtual void addDForce();
    virtual double getPotentialEnergy();

    // -- Temporary added here for matrix ForceField
    void contributeToMatrixDimension(unsigned int * const, unsigned int * const)
    {}
    ;
    void computeMatrix(Sofa::Components::Common::SofaBaseMatrix *, double , double , double, unsigned int &)
    {}
    ;
    void computeVector(Sofa::Components::Common::SofaBaseVector *, unsigned int &)
    {}
    ;
    void matResUpdatePosition(Sofa::Components::Common::SofaBaseVector *, unsigned int & )
    {}
    ;


    // -- VisualModel interface
    void draw();
    void initTextures()
    { }
    ;
    void update()
    { }
    ;

    DataField<int> f_method;
    DataField<Real> f_poisson;
    DataField<Real> f_young;
    DataField<Real> f_damping;

    /*	void setPoissonRatio(Real val) { this->_poissonRatio = val; }
    	void setYoungModulus(Real val) { this->_youngModulus = val; }
    	void setMethod(int val) { this->_method = val; }*/

    Core::MechanicalObject<DataTypes>* getObject()
    {
        return _object;
    }

protected :

    /// f += Kx where K is the stiffness matrix and x a displacement
    virtual void applyStiffness( VecCoord& f, Real h, const VecCoord& x );
    void computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c);
    void computeMaterialStiffnesses();
    void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J );

    ////////////// small displacements method
    void initSmall();
    void accumulateForceSmall( VecCoord& f, const VecCoord & p, Index elementIndex, bool implicit = false );
    void accumulateDampingSmall( VecCoord& f, Index elementIndex );
    void applyStiffnessSmall( VecCoord& f, Real h, const VecCoord& x );

    ////////////// large displacements method
    std::vector< fixed_array <Coord, 3> > _rotatedInitialElements;   ///< The initials positions in its frame
    std::vector< Transformation > _rotations;
    void initLarge();
    void computeRotationLarge( Transformation &r, const VecCoord &p, const Index &a, const Index &b, const Index &c);
    void accumulateForceLarge( VecCoord& f, const VecCoord & p, Index elementIndex, bool implicit=false );
    void accumulateDampingLarge( VecCoord& f, Index elementIndex );
    void applyStiffnessLarge( VecCoord& f, Real h, const VecCoord& x );
};


} // namespace Components


} // namespace Sofa


