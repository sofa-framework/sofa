/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

namespace CGoGN {

namespace Geom {

template<typename PFP>
Frame<PFP>::Frame(const VEC3& X, const VEC3& Y, const VEC3& Z)
{
	const VEC3 refX(Xx,Xy,Xz) ;
	const VEC3 refY(Yx,Yy,Yz) ;
	const VEC3 refZ(Zx,Zy,Zz) ;

	if (!isDirectOrthoNormalFrame<PFP>(X,Y,Z))
		return ;

	REAL& alpha = m_EulerAngles[0] ;
	REAL& beta = m_EulerAngles[1] ;
	REAL& gamma = m_EulerAngles[2] ;

	VEC3 lineOfNodes = refZ ^ Z ;
	if (lineOfNodes.norm2() < 1e-5) // if Z ~= m_Z
	{
		lineOfNodes = refX ; // = reference T
		alpha = 0 ;
		gamma = 0 ;
	}
	else
	{
		lineOfNodes.normalize() ;

		// angle between reference T and line of nodes
		alpha = (refY*lineOfNodes > 0 ? 1 : -1) * std::acos(std::max(std::min(REAL(1.0), refX*lineOfNodes ),REAL(-1.0))) ;
		// angle between reference normal and normal
		gamma = std::acos(std::max(std::min(REAL(1.0), refZ*Z ),REAL(-1.0))) ; // gamma is always positive because the direction of vector lineOfNodes=(reference normal)^(normal) (around which a rotation of angle beta is done later on) changes depending on the side on which they lay w.r.t eachother.
	}
	// angle between line of nodes and T
	beta = (Y*lineOfNodes > 0 ? -1 : 1) * std::acos(std::max(std::min(REAL(1.0), X*lineOfNodes ),REAL(-1.0))) ;
}

template<typename PFP>
Frame<PFP>::Frame(const VEC3& EulerAngles)
{
	m_EulerAngles = EulerAngles ;
}

template<typename PFP>
void Frame<PFP>::getFrame(VEC3& X, VEC3& Y, VEC3& Z) const
{
	const VEC3 refX(Xx,Xy,Xz) ;
	const VEC3 refZ(Zx,Zy,Zz) ;

	// get known data
	const REAL& alpha = m_EulerAngles[0] ;
	const REAL& beta = m_EulerAngles[1] ;
	const REAL& gamma = m_EulerAngles[2] ;

	const VEC3 lineOfNodes = rotate<REAL>(refZ,alpha,refX) ; // rotation around reference normal of vector T
	Z = rotate<REAL>(lineOfNodes,gamma,refZ) ; // rotation around line of nodes of vector N
	X = rotate<REAL>(Z,beta,lineOfNodes) ; // rotation around new normal of vector represented by line of nodes
	Y = Z ^ X ;
}

template<typename PFP>
bool Frame<PFP>::equals(const Geom::Frame<PFP>& lf, REAL epsilon) const
{
	return (m_EulerAngles - lf.m_EulerAngles).norm2() < epsilon ;
}

template<typename PFP>
bool Frame<PFP>::operator==(const Frame<PFP>& lf) const
{
	return this->equals(lf) ;
}

template<typename PFP>
bool Frame<PFP>::operator!=(const Frame<PFP>& lf) const
{
	return !(this->equals(lf)) ;
}

template<typename PFP>
bool isNormalizedFrame(const typename PFP::VEC3& X, const typename PFP::VEC3& Y, const typename PFP::VEC3& Z, typename PFP::REAL epsilon)
{
	return X.isNormalized(epsilon) && Y.isNormalized(epsilon) && Z.isNormalized(epsilon) ;
}

template<typename PFP>
bool isOrthogonalFrame(const typename PFP::VEC3& X, const typename PFP::VEC3& Y, const typename PFP::VEC3& Z, typename PFP::REAL epsilon)
{
	return X.isOrthogonal(Y,epsilon) && X.isOrthogonal(Z,epsilon) && Y.isOrthogonal(Z,epsilon) ;
}

template<typename PFP>
bool isDirectFrame(const typename PFP::VEC3& X, const typename PFP::VEC3& Y, const typename PFP::VEC3& Z, typename PFP::REAL epsilon)
{
	typename PFP::VEC3 new_Y = Z ^ X ;		// direct
	typename PFP::VEC3 diffs = new_Y - Y ;		// differences with existing B
	typename PFP::REAL diffNorm = diffs.norm2() ;	// Norm of this differences vector

	return (diffNorm < epsilon) ;		// Verify that this difference is very small
}

template<typename PFP>
bool isDirectOrthoNormalFrame(const typename PFP::VEC3& X, const typename PFP::VEC3& Y, const typename PFP::VEC3& Z, typename PFP::REAL epsilon)
{
	if (!isNormalizedFrame<PFP>(X,Y,Z,epsilon))
	{
		CGoGNerr << "The Frame you want to create and compress is not normalized" << CGoGNendl ;
		return false ;
	}

	if (!isOrthogonalFrame<PFP>(X,Y,Z,epsilon))
	{
		CGoGNerr << "The Frame you want to create and compress is not orthogonal" << CGoGNendl ;
		return false ;
	}

	if (!isDirectFrame<PFP>(X,Y,Z,epsilon))
	{
		CGoGNerr << "The Frame you want to create and compress is not direct" << CGoGNendl ;
		return false ;
	}

	return true ;
}


template<typename REAL>
Geom::Vector<3,REAL> cartToSpherical (const Geom::Vector<3,REAL>& cart)
{
	Geom::Vector<3,REAL> res ;

	const REAL& x = cart[0] ;
	const REAL& y = cart[1] ;
	const REAL& z = cart[2] ;

	REAL& rho = res[0] ;
	REAL& theta = res[1] ;
	REAL& phi = res[2] ;

	rho = cart.norm() ;
	theta = ((y < 0) ? -1 : 1) * std::acos(x / REAL(sqrt(x*x + y*y)) )  ;
	if (isnan(theta))
		theta = 0.0 ;
	phi = std::asin(z) ;

	return res ;
}

template<typename REAL>
Geom::Vector<3,REAL> sphericalToCart (const Geom::Vector<3,REAL>& sph)
{
	Geom::Vector<3,REAL> res ;

	const REAL& rho = sph[0] ;
	const REAL& theta = sph[1] ;
	const REAL& phi = sph[2] ;

	REAL& x = res[0] ;
	REAL& y = res[1] ;
	REAL& z = res[2] ;

	x = rho*cos(theta)*cos(phi) ;
	y = rho*sin(theta)*cos(phi) ;
	z = rho*sin(phi) ;

	assert(-1.000001 < x && x < 1.000001) ;
	assert(-1.000001 < y && y < 1.000001) ;
	assert(-1.000001 < z && z < 1.000001) ;

	return res ;
}

template <typename REAL>
Geom::Vector<3,REAL> rotate (Geom::Vector<3,REAL> axis, REAL angle, Geom::Vector<3,REAL> vector)
{
	// Algorithm extracted from : http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/ section 5
	axis.normalize() ;

	const REAL& u = axis[0] ;
	const REAL& v = axis[1] ;
	const REAL& w = axis[2] ;

	const REAL& x = vector[0] ;
	const REAL& y = vector[1] ;
	const REAL& z = vector[2] ;

	Geom::Vector<3,REAL> res ;
	REAL& xp = res[0] ;
	REAL& yp = res[1] ;
	REAL& zp = res[2] ;

	const REAL tmp1 = u*x+v*y+w*z ;
	const REAL cos = std::cos(angle) ;
	const REAL sin = std::sin(angle) ;

	xp = u*tmp1*(1-cos) + x*cos+(v*z-w*y)*sin ;
	yp = v*tmp1*(1-cos) + y*cos-(u*z-w*x)*sin ;
	zp = w*tmp1*(1-cos) + z*cos+(u*y-v*x)*sin ;

	return res ;
}

} // Geom

} // CGoGN
