/*
 * sphericalHarmonics.hpp
 *
 *  Created on: Oct 2, 2013
 *      Author: sauvage
 */

namespace CGoGN
{

namespace Utils
{

template <typename Tscalar,typename Tcoef> int SphericalHarmonics<Tscalar,Tcoef>::resolution = -1;
template <typename Tscalar,typename Tcoef> int SphericalHarmonics<Tscalar,Tcoef>::nb_coefs = -1;
template <typename Tscalar,typename Tcoef> unsigned long SphericalHarmonics<Tscalar,Tcoef>::cpt_instances = 0;

template <typename Tscalar,typename Tcoef> Tscalar SphericalHarmonics<Tscalar,Tcoef>::K_tab[(max_resolution+1)*(max_resolution+1)];
template <typename Tscalar,typename Tcoef> Tscalar SphericalHarmonics<Tscalar,Tcoef>::F_tab[(max_resolution+1)*(max_resolution+1)];


/*************************************************************************
construction, destruction and initialization
**************************************************************************/

template <typename Tscalar,typename Tcoef>
SphericalHarmonics<Tscalar,Tcoef>::SphericalHarmonics()
{
	assert ( (nb_coefs > 0) || !" maybe you forgot to call set_level()");
	coefs = new Tcoef[nb_coefs];
	++ cpt_instances;
	for (int i = 0; i < nb_coefs; i++)
		coefs[i] = Tcoef (0);
}

template <typename Tscalar,typename Tcoef>
SphericalHarmonics<Tscalar,Tcoef>::SphericalHarmonics(SphericalHarmonics const & r)
{
	assert ( (nb_coefs > 0) || !" maybe you forgot to call set_level()");
	coefs = new Tcoef[nb_coefs];
	++cpt_instances;
	for (int i = 0; i < nb_coefs; i++)
		coefs[i] = r.coefs[i];
}

template <typename Tscalar,typename Tcoef>
SphericalHarmonics<Tscalar,Tcoef>::~SphericalHarmonics()
{
	delete[] coefs;
	--cpt_instances;
}

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::set_level(int res_level)
{
	assert(res_level >= 0 && res_level < max_resolution);
	assert(cpt_instances == 0);
	resolution = res_level;
	nb_coefs = (resolution + 1) * (resolution + 1);
	init_K_tab();
}

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::set_nb_coefs(int nbc)
{
	assert(nbc > 0);
	int sq = ceil(sqrt(nbc)) ;
	assert(sq*sq == nbc || !"Number of coefs does not fill the last level") ;
	set_level(sq-1) ;
}

/*************************************************************************
evaluation
**************************************************************************/

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::set_eval_direction (Tscalar theta, Tscalar phi)
{
	compute_P_tab(cos(theta));
	compute_y_tab(phi);
}

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::set_eval_direction (Tscalar x, Tscalar y, Tscalar z)
{
	compute_P_tab(z);

	Tscalar phi (0);
	if ((x*x + y*y) > 0.0)
		phi = atan2(y, x);

	compute_y_tab(phi);
}

template <typename Tscalar,typename Tcoef>
Tcoef SphericalHarmonics<Tscalar,Tcoef>::evaluate () const
{
	Tcoef r (0); // (0.0,0.0,0.0); //  TODO : use Tcoef (0)
	for (int i = 0; i < nb_coefs; i++)
	{
		r += coefs[i] * F_tab[i];
	}
	return r;
}

template <typename Tscalar,typename Tcoef>
Tcoef SphericalHarmonics<Tscalar,Tcoef>::evaluate_at (Tscalar theta, Tscalar phi) const
{
	set_eval_direction(theta, phi);
	return evaluate();
}

template <typename Tscalar,typename Tcoef>
Tcoef SphericalHarmonics<Tscalar,Tcoef>::evaluate_at (Tscalar x, Tscalar y, Tscalar z) const
{
	set_eval_direction(x, y, z);
	return evaluate();
}

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::init_K_tab ()
{
	for(int l = 0; l <= resolution; l++)
	{
		// recursive computation of the squares
		K_tab[index(l,0)] = (2*l+1) / (4*M_PI);
		for (int m = 1; m <= l; m++)
			K_tab[index(l,m)] = K_tab[index(l,m-1)] / (l-m+1) / (l+m);
		// square root + symmetry
		K_tab[index(l,0)] = sqrt(K_tab[index(l,0)]);
		for (int m = 1; m <= l; m++)
		{
			K_tab[index(l,m)] = sqrt(K_tab[index(l,m)]);
			K_tab[index(l,-m)] = K_tab[index(l,m)];
		}
	}
}

/* obsolete : was used for shaders
template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::copy_K_tab (Tscalar tab[])
{
	assert ( (nb_coefs>0) || !" maybe you forgot to call set_level()");
	for (unsigned int i = 0 ; i < nb_coefs ; ++i)
		tab[i] = K_tab[i] ;
}
*/

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::compute_P_tab (Tscalar t)
{
//	if (t<0) {t=-t*t;} else {t=t*t;} // for plotting only : expand the param near equator

	F_tab[index(0,0)] = 1;
	for (int l = 1; l <= resolution; l++)
	{
		F_tab[index(l,l)] = (1-2*l) * sqrt(1-t*t) * F_tab[index(l-1,l-1)];  // first diago
		F_tab[index(l,l-1)] = t * (2*l-1) * F_tab[index(l-1,l-1)];// second diago
		for (int m = 0; m <= l-2; m++)
		{// remaining of the line under the 2 diago
			F_tab[index(l,m)] = t * (2*l-1) / (float) (l-m) * F_tab[index(l-1,m)] - (l+m-1) / (float) (l-m) * F_tab[index(l-2,m)];
		}
	}
}

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::compute_y_tab (Tscalar phi)
{
	for (int l = 0; l <= resolution; l++)
	{
		F_tab[index(l,0)] *= K_tab[index(l,0)]; // remove for plotting
	}

	for (int m = 1; m <= resolution; m++)
	{
		Tscalar cos_m_phi = cos ( m * phi );
		Tscalar sin_m_phi = sin ( m * phi );

		for (int l = m; l <= resolution; l++)
		{
			F_tab[index(l,m)] *= sqrt(2.0); // remove for plotting
			F_tab[index(l,m)] *= K_tab[index(l,m)]; // remove for plotting
			F_tab[index(l,-m)] = F_tab[index(l,m)] * sin_m_phi ; // store the values for -m<0 in the upper triangle
			F_tab[index(l,m)] *= cos_m_phi;
		}
	}

}

/*************************************************************************
I/O
**************************************************************************/

template <typename Tscalar,typename Tcoef>
std::ostream & operator << (std::ostream & os, const SphericalHarmonics<Tscalar,Tcoef> & sh)
{
	for (int l = 0; l <= sh.resolution; l++)
	{
		for (int m = -l; m <= l; m++)
			os << sh.get_coef(l,m) << "\t";
		os << std::endl;
	}
	return os;
}

/*************************************************************************
operators
**************************************************************************/

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::operator = (const SphericalHarmonics<Tscalar,Tcoef>& sh)
{
	for (int i = 0; i < nb_coefs; i++)
		this->coefs[i] = sh.coefs[i];
}

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::operator += (const SphericalHarmonics<Tscalar,Tcoef>& sh)
{
	for (int i = 0; i < nb_coefs; i++)
		this->coefs[i] += sh.coefs[i];
}

template <typename Tscalar,typename Tcoef>
SphericalHarmonics<Tscalar,Tcoef> SphericalHarmonics<Tscalar,Tcoef>::operator + (const SphericalHarmonics<Tscalar,Tcoef>& sh) const
{
	SphericalHarmonics<Tscalar,Tcoef> res(*this);
	res += sh;
	return res;
}

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::operator -= (const SphericalHarmonics<Tscalar,Tcoef>& sh)
{
	for (int i = 0; i < nb_coefs; i++)
		this->coefs[i] -= sh.coefs[i];
}

template <typename Tscalar,typename Tcoef>
SphericalHarmonics<Tscalar,Tcoef> SphericalHarmonics<Tscalar,Tcoef>::operator - (const SphericalHarmonics<Tscalar,Tcoef>& sh) const
{
	SphericalHarmonics<Tscalar,Tcoef> res(*this);
	res -= sh;
	return res;
}

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::operator *= (Tscalar s)
{
	for (int i = 0; i < nb_coefs; i++)
		this->coefs[i] *= s;
}

template <typename Tscalar,typename Tcoef>
SphericalHarmonics<Tscalar,Tcoef> SphericalHarmonics<Tscalar,Tcoef>::operator * (Tscalar s) const
{
	SphericalHarmonics<Tscalar,Tcoef> res(*this);
	res *= s;
	return res;
}

template <typename Tscalar,typename Tcoef>
void SphericalHarmonics<Tscalar,Tcoef>::operator /= (Tscalar s)
{
	for (int i = 0; i < nb_coefs; i++)
		this->coefs[i] /= s;
}

template <typename Tscalar,typename Tcoef>
SphericalHarmonics<Tscalar,Tcoef> SphericalHarmonics<Tscalar,Tcoef>::operator / (Tscalar s) const
{
	SphericalHarmonics<Tscalar,Tcoef> res(*this);
	res /= s;
	return res;
}

/*************************************************************************
fitting
**************************************************************************/

template <typename Tscalar,typename Tcoef>
template <typename Tdirection, typename Tchannel>
void SphericalHarmonics<Tscalar,Tcoef>::fit_to_data(
	int n,
	Tdirection* t_theta, Tdirection* t_phi,
	Tchannel* t_R, Tchannel* t_G, Tchannel* t_B,
	double lambda)
{
	Eigen::MatrixXd mM (nb_coefs,n); // matrix with basis function values, evaluated for all directions
	// compute mM
	for (int p = 0; p < n; ++p)
	{
		set_eval_direction(t_theta[p], t_phi[p]);
		for (int i = 0; i < nb_coefs; ++i)
		{
			mM(i,p) = F_tab[i];
		}
	}
	fit_to_data(n, mM, t_R, t_G, t_B, lambda);
}

template <typename Tscalar,typename Tcoef>
template <typename Tdirection, typename Tchannel>
void SphericalHarmonics<Tscalar,Tcoef>::fit_to_data(
	int n,
	Tdirection* t_x, Tdirection* t_y, Tdirection* t_z,
	Tchannel* t_R, Tchannel* t_G, Tchannel* t_B,
	double lambda)
{
	Eigen::MatrixXd mM (nb_coefs,n); // matrix with basis function values, evaluated for all directions
	// compute mM
	for (int p=0; p<n; ++p)
	{
		set_eval_direction(t_x[p],t_y[p],t_z[p]);
		for (int i=0; i<nb_coefs; ++i)
		{
			mM(i,p) = F_tab[i];
		}
	}
	fit_to_data(n, mM, t_R, t_G, t_B, lambda);
}

template <typename Tscalar,typename Tcoef>
template <typename Tchannel>
void SphericalHarmonics<Tscalar,Tcoef>::fit_to_data(
	int n,
	Eigen::MatrixXd& mM,
	Tchannel* t_R, Tchannel* t_G, Tchannel* t_B,
	double lambda)
{
	// fits the data t_R, t_G and t_B, according to our 2013 CGF paper
	// mM contains basis function values, (already) evaluated for all input directions
	// works only for 3 channels

	// allocate the memory
	Eigen::MatrixXd mA (nb_coefs, nb_coefs); // matrix A in linear system AC=B
	Eigen::MatrixXd mB (nb_coefs, 3); // matrix B in linear system AC=B : contains [t_R, t_G, t_B]
	Eigen::MatrixXd mC (nb_coefs, 3); // matrix C (solution) in linear system AC=B : contains the RGB coefs of the resulting SH

	// compute mA
	for (int i = 0; i < nb_coefs; ++i)
	{
		for (int j = 0; j < nb_coefs; ++j)
		{
			mA(i,j) = 0;
			for (int p = 0; p < n; ++p)
			{
				mA(i,j) += mM(i,p) * mM(j,p);
			}
			mA(i,j) *= (1.0-lambda) / n;
		}
	}

	for (int l = 0; l <= resolution; ++l)
	{
		for (int m =- l; m <= l; ++m)
		{
			int i = index(l,m);
			mA(i,i) += lambda * l * (l+1) / (4.0*M_PI);
		}
	}

	// compute mB
	for (int i = 0; i < nb_coefs; ++i)
	{
		mB(i,0) = 0.0;
		mB(i,1) = 0.0;
		mB(i,2) = 0.0;
		for (int p = 0; p < n; ++p)
		{
			mB(i,0) += mM(i,p) * t_R[p];
			mB(i,1) += mM(i,p) * t_G[p];
			mB(i,2) += mM(i,p) * t_B[p];
		}
		mB(i,0) *= (1.0-lambda) / n;
		mB(i,1) *= (1.0-lambda) / n;
		mB(i,2) *= (1.0-lambda) / n;
	}

	// solve the system with LDLT decomposition
	Eigen::LDLT<Eigen::MatrixXd> solver (mA);
	mC = solver.solve(mB);

//	std::cout << "phi = " << mM << std::endl;
//	std::cout << mA << std::endl;
//	std::cout << "*" << std::endl;
//	std::cout << mC << std::endl;
//	std::cout << "=" << std::endl;
//	std::cout << mB << std::endl;

	// store result in the SH
	// it is assumed that Tcoef is VEC3 actually
	for (int i = 0; i < nb_coefs; ++i)
	{
		get_coef(i)[0] = mC(i,0);
		get_coef(i)[1] = mC(i,1);
		get_coef(i)[2] = mC(i,2);
	}
}

} // namespace Utils

} // namespace CGoGN
