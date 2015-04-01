/*
 * spericalHarmonics.h
 *
 *  Created on: Oct 2, 2013
 *      Author: sauvage
 */

#ifndef __SPHERICALHARMONICS_H__
#define __SPHERICALHARMONICS_H__

#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace CGoGN
{

namespace Utils
{

template <typename Tscalar, typename Tcoef>
class SphericalHarmonics
{
private :
	static const int max_resolution = 10 ;                   // max possible resolution for any object of that class
	static int resolution ;                                  // actual resolution level
	static int nb_coefs ;                                    // number of coefs = (resolution + 1) * (resolution + 1)
	static unsigned long cpt_instances;                      // number of instances of the class
	static Tscalar K_tab [(max_resolution+1)*(max_resolution+1)];   // table containing constants K
	static Tscalar F_tab [(max_resolution+1)*(max_resolution+1)];   // table for computing the functions : P or Y or y

	Tcoef* coefs;                                            // table of coefficients

public :
	// construction, destruction and initialization
	SphericalHarmonics();
	SphericalHarmonics(SphericalHarmonics const &);
	~SphericalHarmonics();

	static void set_level(int res_level);
	static void set_nb_coefs (int nb_coefs);
	static int get_resolution () { return resolution; }
	static int get_nb_coefs () { return nb_coefs; }

	// evaluation

	static void set_eval_direction (Tscalar theta, Tscalar phi) ;       // fix the direction in which the SH has to be evaluated
	static void set_eval_direction (Tscalar x, Tscalar y, Tscalar z) ;  // fix the direction in which the SH has to be evaluated
	Tcoef evaluate () const;				  						    // evaluates at a fixed direction

	Tcoef evaluate_at (Tscalar theta, Tscalar phi) const;               // eval spherical coordinates
	Tcoef evaluate_at (Tscalar x, Tscalar y, Tscalar z) const;          // eval cartesian coordinates

	// I/O
	const Tcoef& get_coef (int l, int m) const {assert ((l>=0 && l <=resolution) || !" maybe you forgot to call set_level()"); assert (m >= (-l) && m <= l); return get_coef(index(l,m));}
	Tcoef& get_coef (int l, int m) {assert ((l>=0 && l <=resolution) || !" maybe you forgot to call set_level()"); assert (m >= (-l) && m <= l); return get_coef(index(l,m));}
	template <typename TS,typename TC> friend std::ostream & operator<< (std::ostream & os, const SphericalHarmonics<TS,TC> & sh);

	// operators
	void operator= (const SphericalHarmonics<Tscalar,Tcoef>&);
	void operator+= (const SphericalHarmonics<Tscalar,Tcoef>&);
	SphericalHarmonics<Tscalar,Tcoef> operator+ (const SphericalHarmonics<Tscalar,Tcoef>&) const;
	void operator-= (const SphericalHarmonics<Tscalar,Tcoef> &);
	SphericalHarmonics<Tscalar,Tcoef> operator- (const SphericalHarmonics<Tscalar,Tcoef>&) const;
	void operator*= (Tscalar);
	SphericalHarmonics<Tscalar,Tcoef> operator* (Tscalar) const;
	void operator/= (Tscalar);
	SphericalHarmonics<Tscalar,Tcoef> operator/ (Tscalar) const;

	std::string CGoGNnameOfType() const { return "SphericalHarmonics"; }

//	static void copy_K_tab (Tscalar tab[]) ; // obsolete, was used for shaders

	// fitting
	template <typename Tdirection, typename Tchannel>
	void fit_to_data(int n, Tdirection* t_theta, Tdirection* t_phi, Tchannel* t_R, Tchannel* t_G, Tchannel* t_B, double lambda);
	template <typename Tdirection, typename Tchannel>
	void fit_to_data(int n, Tdirection* t_x, Tdirection* t_y, Tdirection* t_z, Tchannel* t_R, Tchannel* t_G, Tchannel* t_B, double lambda);

private :
	static int index (int l, int m) {return l*(l+1)+m;}

	// evaluation :
	static void init_K_tab (); // compute the normalization constants K_l^m and store them into K_tab
	static void compute_P_tab (Tscalar t); // Compute Legendre Polynomials at parameter t and store them in F_tab (only for m>=0)
	static void compute_y_tab (Tscalar phi); // Compute the real basis functions y_l^m at (theta, phi) and store them in F_tab (compute_P_tab must have been called before)

	const Tcoef& get_coef (int i) const {assert ((i>=0 && i<nb_coefs ) || !" maybe you forgot to call set_level()"); return coefs[i];}
	Tcoef& get_coef (int i) {assert ((i>=0 && i<nb_coefs ) || !" maybe you forgot to call set_level()"); return coefs[i];}

	// fitting
	template <typename Tchannel>
	void fit_to_data(int n, Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& mM, Tchannel* t_R, Tchannel* t_G, Tchannel* t_B, double lambda);
};

} // namespace Utils

} // namespace CGoGN

#include "Utils/sphericalHarmonics.hpp"

#endif /* __SPHERICALHARMONICS_H__ */
