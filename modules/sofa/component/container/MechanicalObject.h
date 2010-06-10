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
#ifndef SOFA_COMPONENT_MECHANICALOBJECT_H
#define SOFA_COMPONENT_MECHANICALOBJECT_H

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/XDataPtr.h>
#include <sofa/core/objectmodel/VDataPtr.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <vector>
#include <assert.h>
#include <fstream>

namespace sofa
{

namespace component
{

namespace container
{

using namespace core::behavior;
using namespace core::objectmodel;
using sofa::defaulttype::Vector3;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class MechanicalObjectInternalData
{
public:
};

template <class DataTypes>
class MechanicalObject : public MechanicalState<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MechanicalObject, DataTypes),SOFA_TEMPLATE(MechanicalState, DataTypes));

    typedef MechanicalState<DataTypes> Inherited;
    typedef typename Inherited::VecId VecId;
    typedef typename Inherited::VMultiOp VMultiOp;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::VecConst VecConst;
    typedef typename defaulttype::SparseConstraint<Deriv> SparseConstraint;
    typedef typename SparseConstraint::const_data_iterator ConstraintIterator;
    typedef typename DataTypes::SparseVecDeriv::const_data_iterator SparseVecDerivIterator;
    typedef typename core::behavior::BaseMechanicalState::ConstraintBlock ConstraintBlock;

protected:
    VecCoord* x;
    VecDeriv* v;
    VecDeriv* f;
    VecDeriv* internalForces;
    VecDeriv* externalForces;
    VecDeriv* dx;
    VecCoord* x0;
    VecCoord* reset_position;

    VecDeriv* v0;
    VecCoord* xfree; // stores the position of the mechanical objet after a free movement (p.e. gravity action)
    VecDeriv* vfree; // stores the velocity of the mechanical objet after a free movement (p.e. gravity action)

    // Constraints stored in the Mechanical State
    // The storage is a SparseMatrix
    // Each constraint (Type TConst) contains the index of the related DOF
    VecConst *c;
    sofa::helper::vector<unsigned int> constraintId;

    bool initialized;
    Data< Vector3 > translation;
    Data< Vector3> rotation;
    Data< Vector3 > scale;
    Data< Vector3 > translation2;
    Data< Vector3> rotation2;
    sofa::core::objectmodel::DataFileName filename;
    Data< bool> ignoreLoader;
    Data<int> f_reserve;

    /// @name Integration-related data
    /// @{

    sofa::helper::vector< VecCoord * > vectorsCoord;
    sofa::helper::vector< VecDeriv * > vectorsDeriv;
    sofa::helper::vector< VecConst * > vectorsConst;
    int vsize; ///< Number of elements to allocate in vectors

    void setVecCoord(unsigned int index, VecCoord* v);
    void setVecDeriv(unsigned int index, VecDeriv* v);
    void setVecConst(unsigned int index, VecConst* v);

#ifdef SOFA_SMP
    sofa::helper::vector< bool > vectorsCoordSharedAllocated;
    sofa::helper::vector< bool > vectorsDerivSharedAllocated;
#endif

    /// @}
    /// Given the numero of a constraint Equation, find the index in the VecConst C, where the constraint is actually stored
    unsigned int getIdxConstraintFromId(unsigned int id) const;

    MechanicalObjectInternalData<DataTypes> data;

    friend class MechanicalObjectInternalData<DataTypes>;

    std::ofstream* m_gnuplotFileX;
    std::ofstream* m_gnuplotFileV;

    class Loader;

public:

    MechanicalObject();
    MechanicalObject& operator = ( const MechanicalObject& );

    virtual ~MechanicalObject();

    virtual bool canPrefetch() const { return false; }

    virtual bool load(const char* filename);

    virtual void parse ( BaseObjectDescription* arg );

    XDataPtr<DataTypes>* const f_X;
    VDataPtr<DataTypes>* const f_V;
    VDataPtr<DataTypes>* const f_F;
    VDataPtr<DataTypes>* const f_externalF;
    VDataPtr<DataTypes>* const f_Dx;
    XDataPtr<DataTypes>* const f_Xfree;
    VDataPtr<DataTypes>* const f_Vfree;

    XDataPtr<DataTypes>* const f_X0;

    Data<SReal> restScale;

    Data<bool> debugViewIndices;
    Data<float> debugViewIndicesScale;

    virtual VecCoord* getX()  { f_X->beginEdit(); return x;  }
    virtual VecDeriv* getV()  { f_V->beginEdit(); return v;  }
    virtual VecDeriv* getF()  { return getVecDeriv(_forceId.index);  }
    virtual VecDeriv* getExternalForces()  { f_externalF->beginEdit(); return externalForces;  }
    virtual VecDeriv* getDx() { f_Dx->beginEdit(); return dx; }
    virtual VecConst* getC() { return c;}
    virtual VecCoord* getXfree() { f_Xfree->beginEdit(); return xfree; }
    virtual VecDeriv* getVfree() { f_Vfree->beginEdit(); return vfree;  }
    VecCoord* getX0() { f_X0->beginEdit(); return x0;}
    virtual VecCoord* getXReset() { return reset_position; }

    virtual const VecCoord* getX()  const { return x;  }
    virtual const VecCoord* getX0()  const { return x0;  }
    virtual const VecDeriv* getV()  const { return v;  }
    virtual const VecDeriv* getV0()  const { return v0;  }
    virtual const VecDeriv* getF()  const { return getVecDeriv(_forceId.index) ;  }
    virtual const VecDeriv* getExternalForces()  const { return externalForces;  }
    virtual const VecDeriv* getDx() const { return dx; }
    virtual const VecConst* getC() const { return c; }
    virtual const VecCoord* getXfree() const { return xfree; }
    virtual const VecDeriv* getVfree()  const { return vfree;  }
    virtual const VecCoord* getXReset() const { return reset_position; }

    virtual void init();
    virtual void reinit();

    virtual void storeResetState();

    virtual void reset();

    virtual void writeX(std::ostream& out);
    virtual void readX(std::istream& in);
    virtual double compareX(std::istream& in);
    virtual void writeV(std::ostream& out);
    virtual void readV(std::istream& in);
    virtual double compareV(std::istream& in);
    virtual void writeF(std::ostream& out);
    virtual void writeDx(std::ostream& out);

    virtual void writeState( std::ostream& out );

    virtual void initGnuplot(const std::string path);
    virtual void exportGnuplot(Real time);

    virtual void resize( int vsize);
    virtual void reserve(int vsize);

    virtual bool addBBox(double* minBBox, double* maxBBox);

    int getSize() const
    {
        return vsize;
    }
    double getPX(int i) const { Real x=0.0,y=0.0,z=0.0; DataTypes::get(x,y,z,(*getX())[i]); return (SReal)x; }
    double getPY(int i) const { Real x=0.0,y=0.0,z=0.0; DataTypes::get(x,y,z,(*getX())[i]); return (SReal)y; }
    double getPZ(int i) const { Real x=0.0,y=0.0,z=0.0; DataTypes::get(x,y,z,(*getX())[i]); return (SReal)z; }

    /** \brief Overwrite values at index outputIndex by the ones at inputIndex.
     *
     */
    void replaceValue (const int inputIndex, const int outputIndex);



    /** \brief Exchange values at indices idx1 and idx2.
     *
     */
    void swapValues (const int idx1, const int idx2);



    /** \brief Reorder values according to parameter.
     *
     * Result of this method is :
     * newValue[ i ] = oldValue[ index[i] ];
     */
    void renumberValues( const sofa::helper::vector<unsigned int> &index );



    /** \brief Replace the value at index by the sum of the ancestors values weithed by the coefs.
     *
     * Sum of the coefs should usually equal to 1.0
     */
    void computeWeightedValue( const unsigned int i, const sofa::helper::vector< unsigned int >& ancestors, const sofa::helper::vector< double >& coefs);

    /** \brief Compute the values attached to a new point.
     *
     */
//	void computeNewPoint( const unsigned int i, const sofa::helper::vector< double >& m_x);

    // Force the position of a point (and force its velocity to zero value)
    void forcePointPosition( const unsigned int i, const sofa::helper::vector< double >& m_x);

    virtual void applyTranslation (const double dx,const double dy,const double dz);

    // rotation using Euler Angles in degree
    virtual void applyRotation (const double rx, const double ry, const double rz);

    virtual void applyRotation (const defaulttype::Quat q);

    virtual void applyScale (const double sx,const double sy,const double sz);

    /// Get the indices of the particles located in the given bounding box
    void getIndicesInSpace(sofa::helper::vector<unsigned>& indices, Real xmin, Real xmax, Real ymin, Real ymax, Real zmin, Real zmax) const;


    /// @Base Matrices and Vectors Interface
    /// @{

    /// Load local mechanical data stored in the state in a (possibly smaller) vector
    virtual void loadInVector(defaulttype::BaseVector *, VecId , unsigned int);

    /// Load local mechanical data stored in the state in a global BaseVector basically stored in solvers
    virtual void loadInBaseVector(defaulttype::BaseVector *, VecId , unsigned int &);

    /// Add data stored in a BaseVector to a local mechanical vector of the MechanicalState
    virtual void addBaseVectorToState(VecId , defaulttype::BaseVector *, unsigned int &);

    /// Add data stored in a Vector (whose size is smaller or equal to the State vector)  to a local mechanical vector of the MechanicalState
    virtual void addVectorToState(VecId , defaulttype::BaseVector *, unsigned int &);

    /// @}



    /// Express the matrix L in term of block of matrices, using the indices of the lines in the VecConst container
    virtual std::list<ConstraintBlock> constraintBlocks( const std::list<unsigned int> &indices) const;
    virtual SReal getConstraintJacobianTimesVecDeriv( unsigned int line, VecId id);


    void setFilename(std::string s) {filename.setValue(s);};

    void setTranslation(double dx, double dy, double dz) {translation.setValue(Vector3(dx,dy,dz));};
    void setRotation(double rx, double ry, double rz) {rotation.setValue(Vector3(rx,ry,rz));};
    void setScale(double sx, double sy, double sz) {scale.setValue(Vector3(sx,sy,sz));};

    virtual Vector3 getTranslation() const {return translation.getValue();};
    virtual Vector3 getRotation() const {return rotation.getValue();};
    virtual Vector3 getScale() const {return scale.getValue();};


    void setIgnoreLoader(bool b) {ignoreLoader.setValue(b);}


    std::string getFilename() {return filename.getValue();};

    virtual void addDxToCollisionModel(void);

    void setConstraintId(unsigned int);
    sofa::helper::vector<unsigned int>& getConstraintId();
    /// Renumber the constraint ids with the given permutation vector
    void renumberConstraintId(const sofa::helper::vector<unsigned>& renumbering);


    /// @name Integration related methods
    /// @{

    virtual void beginIntegration(Real dt);

    virtual void endIntegration(Real dt);

    virtual void accumulateForce();

    VecCoord* getVecCoord(unsigned int index);
    const VecCoord* getVecCoord(unsigned int index) const;

    VecDeriv* getVecDeriv(unsigned int index);
    const VecDeriv* getVecDeriv(unsigned int index) const;

    VecConst* getVecConst(unsigned int index);
    const VecConst* getVecConst(unsigned int index) const;

    virtual void vAvail(VecId& v);

    virtual void vAlloc(VecId v);

    virtual void vFree(VecId v);

#ifdef SOFA_SMP
    virtual void vOp(VecId , VecId  = VecId::null(), VecId  = VecId::null(), double =1.0,a1::Shared<double> * =NULL);
    virtual void vOpMEq(VecId , VecId  = VecId::null(),a1::Shared<double> * =NULL);
    virtual void vDot(a1::Shared<double> *,VecId , VecId );
#endif
    virtual void vOp(VecId v, VecId a = VecId::null(), VecId b = VecId::null(), double f=1.0);

    virtual void vMultiOp(const VMultiOp& ops);

    virtual void vThreshold( VecId a, double threshold );

    virtual double vDot(VecId a, VecId b);

    virtual void setX(VecId v);

    virtual void setXfree(VecId v);

    virtual void setVfree(VecId v);

    virtual void setV(VecId v);

    virtual void setF(VecId v);

    virtual void setDx(VecId v);

    virtual void setC(VecId v);

    virtual void resetForce();

    virtual void resetAcc();

    virtual void resetConstraint();

    virtual sofa::core::VecId getForceId() const { return _forceId; }


    /// @}

    /// @name Debug
    /// @{
    virtual void printDOF( VecId, std::ostream& =std::cerr, int firstIndex=0, int range=-1 ) const ;
    virtual unsigned printDOFWithElapsedTime(VecId, unsigned =0, unsigned =0, std::ostream& =std::cerr );
    //
    void draw();
    /// @}

    // handle state changes
    virtual void handleStateChange();

    /// Find mechanical particles hit by the given ray.
    /// A mechanical particle is defined as a 2D or 3D, position or rigid DOF
    /// Returns false if this object does not support picking
    virtual bool pickParticles(double rayOx, double rayOy, double rayOz, double rayDx, double rayDy, double rayDz, double radius0, double dRadius,
            std::multimap< double, std::pair<sofa::core::behavior::BaseMechanicalState*, int> >& particles);

protected:
    sofa::core::topology::BaseMeshTopology* _topology;
    sofa::core::VecId _forceId;

};

#if defined(WIN32) && !defined(SOFA_COMPONENT_CONTAINER_MECHANICALOBJECT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Vec3dTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Vec2dTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Vec1dTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Vec6dTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Rigid3dTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Vec3fTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Vec2fTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Vec1fTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Vec6fTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Rigid3fTypes>;
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::Rigid2fTypes>;
#endif
extern template class SOFA_COMPONENT_CONTAINER_API MechanicalObject<defaulttype::LaparoscopicRigid3Types>;
#endif

}

} // namespace component

} // namespace sofa

#endif
