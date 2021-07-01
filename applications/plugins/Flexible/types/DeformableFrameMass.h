#ifndef __SOFA_FLEXIBLE_FRAMEMASS_H__
#define __SOFA_FLEXIBLE_FRAMEMASS_H__


#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>

namespace sofa
{

namespace defaulttype
{



    /** Mass associated with a frame */
    template<std::size_t _spatial_dimensions,std::size_t _dim, typename _Real>
    class DeformableFrameMass : public type::Mat<_dim,_dim, _Real>
    {
    public:
        typedef _Real Real;

        static const std::size_t spatial_dimensions = _spatial_dimensions;  ///< Number of dimensions the frame is moving in, typically 3
        static const std::size_t VSize = _dim;

        typedef type::Mat<VSize, VSize, Real> MassMatrix;


        DeformableFrameMass() : MassMatrix(), m_invMassMatrix(NULL)
        {
        }

        /// build a uniform, diagonal matrix
        DeformableFrameMass( Real m ) : MassMatrix(), m_invMassMatrix(NULL)
        {
            setValue( m );
        }


        ~DeformableFrameMass()
        {
            if( m_invMassMatrix )
            {
                delete m_invMassMatrix;
                m_invMassMatrix = NULL;
            }
        }

        /// make a null inertia matrix
        void clear()
        {
            MassMatrix::clear();
            if( m_invMassMatrix ) m_invMassMatrix->clear();
        }

        /// @returns the invert of the mass matrix
        const MassMatrix& getInverse() const
        {
            // optimization: compute the mass invert only once (needed by explicit solvers)
            if( !m_invMassMatrix )
            {
                m_invMassMatrix = new MassMatrix;
                m_invMassMatrix->invert( *this );
            }
            return *m_invMassMatrix;
        }

        /// set a uniform, diagonal matrix
        virtual void setValue( Real m )
        {
            for( unsigned i=0 ; i<VSize ; ++i ) (*this)(i,i) = m;
            updateInverse();
        }


        /// copy
        void operator= ( const MassMatrix& m )
        {
            *((MassMatrix*)this) = m;
            updateInverse();
        }

        /// this += m
        void operator+= ( const MassMatrix& m )
        {
            *((MassMatrix*)this) += m;
            updateInverse();
        }

        /// this -= m
        void operator-= ( const MassMatrix& m )
        {
            *((MassMatrix*)this) -= m;
            updateInverse();
        }

        /// this *= m
        void operator*= ( const MassMatrix& m )
        {
            *((MassMatrix*)this) *= m;
            updateInverse();
        }

        /// apply a factor to the mass matrix
        void operator*= ( Real m )
        {
            *((MassMatrix*)this) *= m;
            updateInverse();
        }

        /// apply a factor to the mass matrix
        void operator/= ( Real m )
        {
            *((MassMatrix*)this) /= m;
            updateInverse();
        }



        /// operator to cast to const Real, supposing the mass is uniform (and so diagonal)
        operator const Real() const
        {
            return (*this)(0,0);
        }


        /// @todo overload these functions so they can be able to update 'm_invMassMatrix' after modifying 'this'
        //void fill(real r)
        // transpose
        // transpose(m)
        // addtranspose
       //subtranspose



    protected:

        mutable MassMatrix *m_invMassMatrix; ///< a pointer to the inverse of the mass matrix

        /// when the mass matrix is changed, if the inverse exists, it has to be updated
        /// @warning there are certainly cases (non overloaded functions or non virtual) that modify 'this' without updating 'm_invMassMatrix'. Another solution = keep a copy of 'this' each time the inversion is done, and when getInverse, if copy!=this -> update
        void updateInverse()
        {
            if( m_invMassMatrix ) m_invMassMatrix->invert( *this );
        }
    };

    template<class Deriv,std::size_t _spatial_dimensions,std::size_t _dim,typename _Real>
    Deriv operator/(const Deriv& d, const DeformableFrameMass<_spatial_dimensions, _dim, _Real>& m)
    {
        return m.getInverse() * d;
    }

    template<class Deriv,std::size_t _spatial_dimensions,std::size_t _dim,typename _Real>
    Deriv operator*(const DeformableFrameMass<_spatial_dimensions, _dim,_Real>& m,const Deriv& d)
    {
        return d * m;
    }



} // namespace defaulttype

} // namespace sofa

#endif //__SOFA_FLEXIBLE_FRAMEMASS_H__
