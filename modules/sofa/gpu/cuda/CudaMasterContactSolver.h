#ifndef SOFA_COMPONENT_ODESOLVER_CUDAMASTERCONTACTSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_CUDAMASTERCONTACTSOLVER_H

#include <sofa/simulation/tree/MasterSolverImpl.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>
#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/gpu/cuda/CudaLCP.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::component::linearsolver;
using namespace helper::system::thread;
using namespace sofa::gpu::cuda;
/*
class CudaLPtrFullMatrix : public FullMatrix<double> {
	public :

		CudaMatrix<float>& getCudaMatrix() {
			for (unsigned j=0; j<m.getSizeX(); j++) {
				for (unsigned i=0; i<m.getSizeY(); i++) {
					m[i][j] = this->FullMatrix<double>::operator [](i)[j];
				}
			}
			return m;
		}

		void resize(int nbRow, int nbCol) {
			m.resize(nbCol,nbRow,warp_size);
			this->FullMatrix<double>::resize(nbRow,nbCol);
		}

		void setwarpsize(double mu) {
			if (mu>0.0) warp_size = 96;
			else warp_size = 64;
		}

	private :
		CudaMatrix<float> m;
		int warp_size;
};

class CudaFullVector : public FullVector<double> {

	public :
		CudaVector<float>& getCudaVector() {
			for (int i=0; i<size(); i++) v[i]=this->FullVector<double>::operator [](i);
			return v;
		}

		void resize(int nbRow) {
			v.resize(nbRow);
			this->FullVector<double>::resize(nbRow);
		}

	private :
		CudaVector<float> v;
};
*/

class CudaLPtrFullMatrix : public BaseMatrix
{
public :

    CudaMatrix<float>& getCudaMatrix()
    {
        return m;
    }

    void resize(int nbRow, int nbCol)
    {
        m.resize(nbCol,nbRow,warp_size);
        this->clear();
    }

    void setwarpsize(double mu)
    {
        if (mu>0.0) warp_size = 96;
        else warp_size = 64;
    }

    int rowSize() const
    {
        return m.getSizeY();
    }

    int colSize() const
    {
        return m.getSizeX();
    }

    double element(int i, int j) const
    {
        return m[i][j];
    }

    void clear()
    {
        for (unsigned j=0; j<m.getSizeX(); j++)
        {
            for (unsigned i=0; i<m.getSizeY(); i++)
            {
                m[i][j] = 0.0;
            }
        }
    }

    void set(int i, int j, double v)
    {
        m[i][j] = v;
    }

    void add(int i, int j, double v)
    {
        m[i][j] += v;
    }

private :
    CudaMatrix<float> m;
    int warp_size;
};

class CudaFullVector : public BaseVector
{

public :
    CudaVector<float>& getCudaVector()
    {
        return v;
    }

    float & operator[](int i)
    {
        return v[i];
    }

    const float & operator[](int i) const
    {
        return v[i];
    }

    void resize(int nbRow)
    {
        v.resize(nbRow);
    }

    int size() const
    {
        return v.size();
    }

    double element(int i) const
    {
        return v[i];
    }

    void clear()
    {
        for (int i=0; i<size(); i++) v[i]=0.0;
    }

    void set(int i, double val)
    {
        v[i] = (float) val;
    }

    void add(int i, double val)
    {
        v[i] += (float)val;
    }

private :
    CudaVector<float> v;
};

class CudaMechanicalGetConstraintValueVisitor : public simulation::tree::MechanicalVisitor
{
public:
    CudaMechanicalGetConstraintValueVisitor(defaulttype::BaseVector * v): _v(v) {}

    virtual Result fwdConstraint(simulation::tree::GNode*,core::componentmodel::behavior::BaseConstraint* c)
    {
        c->getConstraintValue(_v);
        return RESULT_CONTINUE;
    }
private:
    defaulttype::BaseVector * _v;
};

class CudaMechanicalGetContactIDVisitor : public simulation::tree::MechanicalVisitor
{
public:
    CudaMechanicalGetContactIDVisitor(long *id, unsigned int offset = 0)
        : _id(id),_offset(offset) {}

    virtual Result fwdConstraint(simulation::tree::GNode*,core::componentmodel::behavior::BaseConstraint* c)
    {
        c->getConstraintId(_id, _offset);
        return RESULT_CONTINUE;
    }

private:
    long *_id;
    unsigned int _offset;
};

class CudaMasterContactSolver : public sofa::simulation::tree::MasterSolverImpl
{
public:
    Data<bool> initial_guess_d;
    /*
    Data < double > tol_d;
    Data < int > maxIt_d;
    Data < double > mu_d;
    */
    Data<int> useGPU_d;

    CudaMasterContactSolver();

    void step (double dt);

    virtual void init();

private:
    std::vector<core::componentmodel::behavior::BaseConstraintCorrection*> constraintCorrections;
    void computeInitialGuess();
    void keepContactForcesValue();

    void build_LCP();

    CudaLPtrFullMatrix _W, _A;
    CudaFullVector _dFree, _f, _res;

    unsigned int _numConstraints;
    double _mu;
    simulation::tree::GNode *context;

    typedef struct
    {
        Vector3 n;
        Vector3 t;
        Vector3 s;
        Vector3 F;
        long id;

    } contactBuf;

    contactBuf *_PreviousContactList;
    unsigned int _numPreviousContact;
    long *_cont_id_list;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
