#ifndef SOFA_CORE_MECHANICALOBJECT_H
#define SOFA_CORE_MECHANICALOBJECT_H

#include "MechanicalModel.h"
#include <vector>
#include <assert.h>

namespace Sofa
{

namespace Core
{

template <class DataTypes>
class MechanicalObject : public MechanicalModel<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

protected:
    VecCoord* x;
    VecDeriv* v;
    VecDeriv* f;
    VecDeriv* dx;
    VecCoord* x0;
    VecDeriv* v0;
    VecDeriv* internalForces;
    VecDeriv* externalForces;
    double translation[3];
    double scale;

    /// @name Integration-related data
    /// @{

    std::vector< VecCoord * > vectorsCoord;
    std::vector< VecDeriv * > vectorsDeriv;
    int vsize; ///< Number of elements to allocate in vectors

    void setVecCoord(unsigned int index, VecCoord* v);
    void setVecDeriv(unsigned int index, VecDeriv* v);

    /// @}

public:
    MechanicalObject();

    virtual ~MechanicalObject();

    VecCoord* getX()  { return x;  }
    VecDeriv* getV()  { return v;  }
    VecDeriv* getF()  { return f;  }
    VecDeriv* getDx() { return dx; }

    const VecCoord* getX()  const { return x;  }
    const VecCoord* getX0()  const { return x0;  }
    const VecDeriv* getV()  const { return v;  }
    const VecDeriv* getV0()  const { return v0;  }
    const VecDeriv* getF()  const { return f;  }
    const VecDeriv* getDx() const { return dx; }

    virtual void init();

    virtual void reset();

    virtual void writeState( std::ostream& out );

    virtual void resize( int vsize);

    int getSize() const
    {
        return vsize;
    }

    /// performs x[outPutIndex]=x[inputIndex]
    void replaceValue (const int inputIndex, const int outputIndex);

    void applyTranslation (double dx, double dy, double dz);

    void applyScale (double s);

    /// Get the indices of the particles located in the given bounding box
    void getIndicesInSpace(std::vector<unsigned>& indices, Real xmin, Real xmax, Real ymin, Real ymax, Real zmin, Real zmax) const;

    /// @name Integration related methods
    /// @{

    virtual void beginIntegration(double dt);

    virtual void endIntegration(double dt);

    virtual void accumulateForce();

    VecCoord* getVecCoord(unsigned int index);

    VecDeriv* getVecDeriv(unsigned int index);

    virtual void vAlloc(VecId v);

    virtual void vFree(VecId v);

    virtual void vOp(VecId v, VecId a = VecId::null(), VecId b = VecId::null(), double f=1.0);

    virtual double vDot(VecId a, VecId b);

    virtual void setX(VecId v);

    virtual void setV(VecId v);

    virtual void setF(VecId v);

    virtual void setDx(VecId v);

    virtual void resetForce();

    /// @}

    /// @name Debug
    /// @{
    virtual void printDOF( VecId, std::ostream& =std::cerr );
    /// @}

    /*! \fn void addBBox()
    *  \brief Used to add the bounding-box of this mechanical model to the given bounding box.
    *
    *  Note that if it does not make sense for this particular object (such as if the DOFs are not 3D), then the default implementation displays a warning message and returns false.
    */
    //virtual bool addBBox(double* minBBox, double* maxBBox);
};

} // namespace Core

} // namespace Sofa

#endif
