/******************************************************************************
*									      *
*			Ho Lab - IoTouch Project			      *
*		       Developer: Nguyen Huu Nhan                             *
*                                  					      *
******************************************************************************/
#pragma once

#include <SofaMiscFem/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaBaseTopology/TopologyData.h>

#include <map>
#include <sofa/helper/map.h>

namespace sofa::component::forcefield
{

template<class DataTypes>
class QuadBendingFEMForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(QuadBendingFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::BaseMeshTopology::Index Index;
    typedef sofa::core::topology::BaseMeshTopology::Quad Element;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads VecElement;
    typedef sofa::core::topology::BaseMeshTopology::QuadsAroundVertex QuadsAroundVertex;

    typedef sofa::helper::Quater<Real> Quat;

    enum {
        //LARGE = 0,   ///< Symbol of small displacements quad solver
        SMALL = 1,   ///< Symbol of large displacements quad solver
    };

protected:

    //bool _anisotropicMaterial;			                 	    /// used to turn on / off optimizations
    typedef defaulttype::Vec<20, Real> Displacement;					    ///< the displacement vector
    typedef defaulttype::Mat<8, 8, Real> MaterialStiffness;				    ///< the matrix of material stiffness
    typedef sofa::helper::vector<MaterialStiffness> VecMaterialStiffness;   ///< a vector of material stiffness matrices
    typedef defaulttype::Mat<32, 20, Real> StrainDisplacement;				    ///< the strain-displacement matrix
    typedef defaulttype::Mat<20, 20, Real> Stiffness;					    ///< the stiffness matrix
    typedef sofa::helper::vector<StrainDisplacement> VecStrainDisplacement; ///< a vector of strain-displacement matrices
    //typedef defaulttype::Mat<3, 3, Real > Transformation;				    ///< matrix for rigid transformations like rotations
    
protected:
    /// ForceField API
    QuadBendingFEMForceField();

    ~QuadBendingFEMForceField() override;
public:
    void init() override;
    void reinit() override;
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;
/// Class to store FEM information on each quad, for topology modification handling
    class QuadInformation
    {
    public:
        /// material stiffness matrices of each quad including Bending and shear component
        MaterialStiffness BendingmaterialMatrix;
        MaterialStiffness ShearmaterialMatrix;
        ///< the strain-displacement matrices vector
        StrainDisplacement strainDisplacementMatrix;
        ///< the stiffness matrix
        Stiffness stiffness;
        Stiffness Bendingstiffness;
        Stiffness Shearstiffness;
        //Real area;
        // large displacement method
        helper::fixed_array<Coord,3> InitialPosElements;
        Coord IntlengthElement;
        Coord IntheightElement;
        Coord Intcentroid;
        Real Inthalflength;
        Real Inthalfheight;

        //helper::vector<Coord> lastNStressDirection;

        QuadInformation() { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const QuadInformation& /*ti*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, QuadInformation& /*ti*/ )
        {
            return in;
        }

        Real differenceToCriteria;
    };
    
/// Class to store FEM information on each edge, for topology modification handling
    class EdgeInformation
    {
    public:
        EdgeInformation(){}
           /* :fracturable(false) {}

        bool fracturable;*/

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeInformation& /*ei*/ )
        {
            return in;
        }
    };

    /// Class to store FEM information on each vertex, for topology modification handling
    class VertexInformation
    {
    public:
        VertexInformation(){}
           /* :sumEigenValues(0.0), stress(0.0) {}

        Coord meanStrainDirection;
        double sumEigenValues;
        Transformation rotation;

        double stress; //average stress of quads around (used only for drawing)*/

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const VertexInformation& /*vi*/)
        {
            return os;
        }
        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, VertexInformation& /*vi*/)
        {
            return in;
        }
    };

/// Topology Data
    topology::QuadData<sofa::helper::vector<QuadInformation> > quadInfo;
    topology::PointData<sofa::helper::vector<VertexInformation> > vertexInfo; ///< Internal point data
    topology::EdgeData<sofa::helper::vector<EdgeInformation> > edgeInfo; ///< Internal edge data

    class QuadHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Quad,helper::vector<QuadInformation> >
    {
    public:
        QuadHandler(QuadBendingFEMForceField<DataTypes>* _ff, topology::QuadData<sofa::helper::vector<QuadInformation> >* _data) : 
        topology::TopologyDataHandler<core::topology::BaseMeshTopology::Quad, sofa::helper::vector<QuadInformation> >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int quadIndex, QuadInformation& ,
                const core::topology::BaseMeshTopology::Quad & t,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);
    protected:
        QuadBendingFEMForceField<DataTypes>* ff;
    };

    sofa::core::topology::BaseMeshTopology* m_topology;
    
    /// Get/Set methods
    Real getPoisson() { return (f_poisson.getValue())[0]; }
    void setPoisson(Real val)
    {
        helper::vector<Real> newP(1, val);
        f_poisson.setValue(newP);
    }
    Real getYoung() { return (f_young.getValue())[0]; }
    void setYoung(Real val)
    {
        helper::vector<Real> newY(1, val);
        f_young.setValue(newY);
    }
    /*Real getDamping() { return f_damping.getValue(); }
    void setDamping(Real val) { f_damping.setValue(val); }*/
    int  getMethod() { return method; }
    void setMethod(int val) { method = val; }
    void setMethod(const std::string& methodName); 
protected : 
    /// Forcefiled computations
    void computeDisplacementSmall(Displacement &D, Index elementIndex, const VecCoord &p);
    void computeBendingStrainDisplacement(StrainDisplacement &Jb, /*Index elementIndex,*/ float gauss1, float gauss2, float l, float h);
    void computeShearStrainDisplacement(StrainDisplacement &Js, /*Index elementIndex,*/ float l, float h);
    void computeElementStiffness( Stiffness &K, Index elementIndex);
    void computeForce(Displacement &F, Index elementIndex, Displacement &D);
    
    virtual void applyStiffness( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );
    virtual void computeBendingMaterialStiffness(int i, Index& a, Index& b, Index& c, Index& d);
    virtual void computeShearMaterialStiffness(int i, Index& a, Index& b, Index& c, Index& d);
   
    ////////////// small displacements method
    void initSmall(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceSmall( VecCoord& f, const VecCoord & p, Index elementIndex);
    //void accumulateDampingSmall( VecCoord& f, Index elementIndex );
    void applyStiffnessSmall( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );
    
    ////////////// large displacements method
    /*void initLarge(int i, Index&a, Index&b, Index&c, Index&d);
    void accumulateForceLarge( VecCoord& f, const VecCoord & p, Index elementIndex);
    void applyStiffnessLarge( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );*/
    
    /*bool updateMatrix;
    int lastFracturedEdgeIndex;*/

public:

    /// Forcefield intern paramaters
    int method;
    Data<std::string> f_method; ///< large: large displacements, small: small displacements
    Data<helper::vector<Real> > f_poisson; ///< Poisson ratio in Hooke's law (vector)
    Data<helper::vector<Real> > f_young; ///< Young modulus in Hooke's law (vector)
    Data<Real> f_thickness;
    QuadHandler* quadHandler;

    /// Link to be set to the topology container in the component graph.
    SingleLink<QuadBendingFEMForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
    
};
#if  !defined(SOFA_COMPONENT_FORCEFIELD_QUADBENDINGFEMFORCEFIELD_CPP)

extern template class SOFA_MISC_FEM_API QuadBendingFEMForceField<defaulttype::Vec3Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_QUADBENDINGFEMFORCEFIELD_CPP)

} // namespace sofa::component::forcefield
