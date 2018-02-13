#ifndef RIGIDMASS_H
#define RIGIDMASS_H

#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/MechanicalState.h>

#include "../utils/se3.h"
#include "../utils/map.h"

#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace component
{

namespace mass
{

/**
   An actual rigid mass matrix. It applies to center-of-mass-centered,
   principal-axis-aligned rigid frames. 

   It is still unclear to me whether the defaulttype::RigidMass
   class/UniformMass/DiagonalMass accounts for frame changes
   correctly, so i rolled my own.

   Since SOFA uses absolute frame for rotational velocities
   (i.e. spatial velocities), the corresponding spatial inertia
   tensors are Is = R.Ib.R^T, where Ib is the body-fixed inertia
   tensor. It seems that Ib is used as a spatial inertia tensor, but I
   might be wrong.

   @author: maxime.tournier@inria.fr
 */

template <class DataTypes>
class RigidMass : public core::behavior::Mass<DataTypes>
{
public:
	SOFA_CLASS(SOFA_TEMPLATE(RigidMass, DataTypes), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));
	
	typedef typename DataTypes::Real real;
	typedef helper::vector<real> mass_type;
	typedef helper::vector< defaulttype::Vec<3, real> > inertia_type;
	
	Data<mass_type> mass; ///< mass of each rigid body
	Data<inertia_type> inertia; ///< inertia of each rigid body
	Data<bool> inertia_forces; ///< compute (explicit) inertia forces
    Data<bool> _draw; ///< debug drawing of the inertia matrix
	
	typedef SE3<real> se3;

	RigidMass() 
		: mass(initData(&mass, "mass", "mass of each rigid body")),
		  inertia(initData(&inertia, "inertia", "inertia of each rigid body")),
          inertia_forces(initData(&inertia_forces, false, "inertia_forces", "compute (explicit) inertia forces")),
          _draw(initData(&_draw, false, "draw", "debug drawing of the inertia matrix"))
    {
        _draw.setGroup("Visualization");
	}
	
protected:
	
	// clamps an index to the largest index in mass/inertia vectors
	unsigned clamp(unsigned i) const {
		return std::min<unsigned>(i, mass.getValue().size() - 1);
	}

    template<class T>
    static T& edit(const Data<T>& data) {
        // hell yeah
        return const_cast<T&>(data.getValue());
    }
    
public:

	void init() {
		this->core::behavior::Mass<DataTypes>::init();

        typedef std::runtime_error error;

        try{ 
            if( !this->mstate )  {
                throw error("no mstate !");
            }

            if( !mass.getValue().size() ) {
                edit(mass).resize( this->mstate->getSize() );
                serr << "empty data 'mass', auto-resizing" << sendl;
            }
            
            if( mass.getValue().size() != inertia.getValue().size() ) {
                edit(inertia).resize(mass.getValue().size());
                serr << "'mass' and 'inertia' data must have the same size, auto-resizing" << sendl;
            }
        } catch(error& e) {
            serr << e.what() << sendl;
            throw e;
        }

		this->reinit();
	} 
	
	typedef typename DataTypes::VecCoord VecCoord;
	typedef typename DataTypes::VecDeriv VecDeriv;
	
	typedef core::objectmodel::Data<VecCoord> DataVecCoord;
	typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

#ifndef SOFA_NO_OPENGL
	void draw(const core::visual::VisualParams* vparams) {
		
        if ( !vparams->displayFlags().getShowBehaviorModels() || !_draw.getValue() )
            return;
        helper::ReadAccessor<VecCoord> x = this->mstate->read(core::ConstVecCoordId::position())->getValue();


        for(unsigned i = 0, n = x.size(); i < n; ++i) {
            const unsigned index = clamp(i);

            const real& m00 = inertia.getValue()[index][0];
            const real& m11 = inertia.getValue()[index][1];
            const real& m22 = inertia.getValue()[index][2];
			
            defaulttype::Vec3d len;
            len[0] = std::sqrt(m11+m22-m00);
            len[1] = std::sqrt(m00+m22-m11);
            len[2] = std::sqrt(m00+m11-m22);

#ifndef SOFA_NO_OPENGL
            helper::gl::Axis::draw(x[i].getCenter(), x[i].getOrientation(), len);
#endif
        }
		
	}
#endif

	void addForce(const core::MechanicalParams* , 
	              DataVecDeriv& _f, 
	              const DataVecCoord& _x, const DataVecDeriv& _v) {

		helper::WriteAccessor< DataVecDeriv > f(_f);
		helper::ReadAccessor< DataVecCoord >  x(_x);
		helper::ReadAccessor< DataVecDeriv >  v(_v);

		typename se3::vec3 g = SE3<SReal>::map(this->getContext()->getGravity()).template cast<real>();

		for(unsigned i = 0, n = this->mstate->getSize(); i < n; ++i) {
			const unsigned index = clamp(i);
		
			se3::map( f[i].getVCenter() ).template head<3>() += mass.getValue()[index] * g;
			
			if( inertia_forces.getValue() ) {
				// explicit inertia, based on usual formula
				// see http://en.wikipedia.org/wiki/Newton-Euler_equations
				typename se3::mat33 R = se3::rotation( x[i] ).toRotationMatrix();

				// spatial velocity
				typename se3::vec3 omega = se3::map( v[i].getAngular() );

				// body inertia tensor
				typename se3::vec3 I = se3::map( inertia.getValue()[ index ]);
					
				// mass
				// SReal m = mass.getValue()[ index ];
					
				se3::map( f[i].getAngular() ) -= omega.cross( R * I.cwiseProduct( R.transpose() * omega) );
				// se3::map( f[i].getLinear() ) -= m * omega.cross( se3::map(v[i].getLinear()) );
			}

		}
		
	
	}

	// perdu: il faut aussi la position pour connaître l'énergie
	// cinétique d'un rigide, sauf si on considère les vitesses en
	// coordonnées locales (ce que sofa ne fait pas). du coup on tape
	// dans this->mstate->getX pour l'obtenir mais l'api devrait gérer ca
    SReal getKineticEnergy( const core::MechanicalParams*,
	                         const DataVecDeriv& _v  ) const {
		helper::ReadAccessor< DataVecDeriv >  v(_v);
		
        SReal res = 0;

		for(unsigned i = 0, n = v.size(); i < n; ++i) {
			const unsigned index = clamp(i);
			
			// body-fixed velocity
			typename se3::vec3 omega_body = se3::rotation( this->mstate->read(core::ConstVecCoordId::position())->getValue()[i] ).inverse() * 
				se3::map(v[i].getVOrientation());
			
			res += 
				mass.getValue()[index] * v[i].getVCenter().norm2() +
				se3::map(inertia.getValue()[index]).cwiseProduct( omega_body ).dot(omega_body);
		}
		
		res *= 0.5;
		
		return res;
	}

	// TODO maybe sign is wrong 
    SReal getPotentialEnergy( const core::MechanicalParams*,
	                           const DataVecCoord& _x  ) const {
		helper::ReadAccessor< DataVecCoord >  x(_x);
				
		defaulttype::Vec3d g ( this->getContext()->getGravity() );

        SReal res = 0;

		for(unsigned i = 0, n = x.size(); i < n; ++i) {
			const unsigned index = clamp(i);
	
			res -= mass.getValue()[index] * (g * x[i].getCenter()); 
		}
		
		return res;
	}


    virtual void addMDx(const core::MechanicalParams* ,
	                    DataVecDeriv& _f, 
	                    const DataVecDeriv& _dx, 
                        SReal factor) {
        helper::WriteAccessor< DataVecDeriv > f(_f);
        helper::ReadAccessor< DataVecDeriv > dx(_dx);

		for(unsigned i = 0, n = this->mstate->getSize(); i < n; ++i) {
			const unsigned index = clamp(i);
			
			using namespace utils;
			
            se3::map(f[i].getLinear()) += (factor * mass.getValue()[ index ]) * se3::map(dx[i].getLinear());

			typename se3::quat q = se3::rotation( this->mstate->read(core::ConstVecCoordId::position())->getValue()[i] );
            se3::map(f[i].getAngular()) += factor *
                ( q * se3::map(inertia.getValue()[ index ]).cwiseProduct( q.conjugate() * se3::map(dx[i].getAngular() ) ));
			
		}
		
	}


    virtual void addMToMatrix(const core::MechanicalParams* mparams,
	                          const sofa::core::behavior::MultiMatrixAccessor* matrix) {
		
		sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix( this->mstate );
		const unsigned size = defaulttype::DataTypeInfo<typename DataTypes::Deriv>::size();

        real mFactor = (real)mparams->mFactorIncludingRayleighDamping(this->rayleighMass.getValue());
		
		for(unsigned i = 0, n = this->mstate->getSize(); i < n; ++i) {

			const unsigned index = clamp(i);
			
			// translation
			for(unsigned j = 0; j < 3; ++j) {
				r.matrix->add(r.offset + size * i + j,
				              r.offset + size * i + j,
                              mass.getValue()[ index ] * mFactor );
			}			              
			
			typename se3::mat33 R = se3::rotation( this->mstate->read(core::ConstVecCoordId::position())->getValue()[i] ).toRotationMatrix();
			
			typename se3::mat33 chunk = R * se3::map( inertia.getValue()[ index ] ).asDiagonal() * R.transpose();
			
			// rotation
			for(unsigned j = 0; j < 3; ++j) {
				for(unsigned k = 0; k < 3; ++k) {
					
					r.matrix->add(r.offset + size * i + 3 + j,
					              r.offset + size * i + 3 + k,
                                  chunk(j, k) * mFactor );
				}
			}			              
			
		}
		
    }

};

}
}
}

#endif
