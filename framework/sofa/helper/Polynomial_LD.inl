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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_POLYNOMIAL_LD_INL
#define SOFA_HELPER_POLYNOMIAL_LD_INL

#include "Polynomial_LD.h"
#include <sstream>
#include <iterator>


namespace sofa
{
namespace helper
{


using namespace sofa::defaulttype;
using namespace std;
/**
 * Tools used in FEM computing
 */

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N>::Monomial_LD()
{
    coef=(Real) 0;
    for(unsigned int i=0; i<N; i++)
    {
        powers[i]=0;
        ostringstream oss; oss << 'l' << i ;
        variables[i]=oss;
    }
}
////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N>::Monomial_LD(const Monomial_LD<Real,N> & b)
{
    coef=b.coef;
    for(unsigned int i=0; i<N; i++)
    {
        powers[i]=b.powers[i];
        variables[i]=b.variables[i];
    }
}
////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N>::Monomial_LD(const Real & m_coef, ...)
{
    coef=m_coef;
    va_list vl;
    va_start(vl,m_coef);
    for(unsigned int i=0; i<N; i++)
    {
        powers[i]=va_arg(vl,int);
        ostringstream oss; oss << 'l' << i ;
        variables[i]=oss;
    }
    va_end(vl);
}
////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N>& Monomial_LD<Real,N>::operator=(const Monomial_LD<Real,N> & b)
{
    coef=b.coef;
    for( unsigned int i=0; i<N; i++)
    {
        powers[i]=b.powers[i];
        variables[i]=b.variables[i];
    }
    return *this;
}
////////////////////////////////

template<typename Real, unsigned int N>
void Monomial_LD<Real,N>::Set(const Real & m_coef, ...)
{
    coef=m_coef;
    va_list vl;
    va_start(vl,m_coef);
    for(unsigned int i=0; i<N; i++)
    {
        powers[i]=va_arg(vl,int);
    }
    va_end(vl);
}
////////////////////////////////
template<typename Real, unsigned int N>
int Monomial_LD<Real,N>::degree()
{
    int degree=0;
    for(unsigned int i=0; i<N; i++)
        degree+=powers[i];
    return degree;
}
////////////////////////////////
template<typename Real, unsigned int N>
bool Monomial_LD<Real,N>::operator==(const Monomial_LD<Real,N> & b) const
{
    bool compare=true;
    if ((coef != b.coef) || (variables.size() != b.variables.size())) compare=false;
    for(unsigned int i=0; i<N; i++)
    {
        if (powers[i] != b.powers[i]) compare=false;
    }
    return compare;
}
////////////////////////////////
template<typename Real, unsigned int N>
bool Monomial_LD<Real,N>::isSamePowers(const Monomial_LD<Real,N> & b) const
{
    bool compare=true;
    if ( variables.size() != b.variables.size() ) compare=false;
    for(unsigned int i=0; i<N; i++)
    {
        if (powers[i] != b.powers[i]) compare=false;
    }
    return compare;
}
////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N> & Monomial_LD<Real,N>::operator+=(const Monomial_LD<Real,N> & b)
{
    if (this->isSamePowers(b))
    {
        this->coef+=b.coef;
    }
    else
    {
        cout<<"WARNING : "<<(*this)<<" + "<<b
            <<"   Not permissed with different powers"<<endl;
    }
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N> & Monomial_LD<Real,N>::operator-=(const Monomial_LD<Real,N> & b)
{
    if (this->isSamePowers(b))
    {
        this->coef-=b.coef;
    }
    else
    {
        cout<<"WARNING : "<<(*this)<<" - "<<b
            <<"   Not permissed with different powers"<<endl;
    }
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N> & Monomial_LD<Real,N>::operator*=(const Monomial_LD<Real,N> & b)
{
    coef*=b.coef;
    for(unsigned int i=0; i<N; i++)
        powers[i] += b.powers[i];
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Real Monomial_LD<Real,N>::operator()(const vector<Real> & x) const
{
    if (x.size()!= N) cout<<"WARNING : value assigned to the monome has not the good number of variable"<<endl;
    Real value= coef;
    for( unsigned int i=0; i<N; i++)
    {
        value*=(Real) pow(x[i],powers[i]);
    }
    return value;
}
////////////////////////////////
template<typename Real, unsigned int N>
Real Monomial_LD<Real,N>::operator()(const RNpoint & x) const
{
    if (x.size()!= N) cout<<"WARNING : value assigned to the monome has not the good number of variable"<<endl;
    Real value= coef;
    for( unsigned int i=0; i<N; i++)
    {
        value*=(Real) pow(x[i],powers[i]);
    }
    return value;
}
////////////////////////////////
template<typename Real, unsigned int N>
Real Monomial_LD<Real,N>::operator()(const vector<Real> & x,unsigned int ideriv) const
{
    //assert( (x.size()==N) && (ideriv < N) );
    if (x.size()!= N) cout<<"WARNING : value assigned to the monome has not the good number of variable"<<endl;
    Real value= coef;

    if (ideriv >= N)
    {
        cout<<"WARNING : "<<ideriv<<"-th derivative couldn't take place for the monomial of:"<<N<<"-variables"<<endl
            <<(*this)<<endl
            <<"CONDITION: id_derivative = { 0,1... (NbVariable-1) }"<<endl<<endl;
    }
    else
    {
        if(ideriv==0)
        {
            value=this->operator()(x);
        }
        else
        {
            for(unsigned int i=0; i<N; i++)
            {
                if (i==ideriv)
                {
                    value*=(Real) powers[i];//derivate
                    value*=(Real) pow(x[i],(powers[i]-1));
                }
                else
                {
                    value*=(Real) pow(x[i],powers[i]);
                }
            }
        }
    }
    return value;
}
////////////////////////////////
template<typename Real, unsigned int N>
Real Monomial_LD<Real,N>::operator()(const RNpoint & x,unsigned int ideriv) const
{
    //assert( (x.size()==N) && (ideriv < N) );
    if (x.size()!= N) cout<<"WARNING : value assigned to the monome has not the good number of variable"<<endl;
    Real value= coef;

    if (ideriv >= N)
    {
        cout<<"WARNING : "<<ideriv<<"-th derivative couldn't take place for the monomial of:"<<N<<"-variables"<<endl
            <<(*this)<<endl
            <<"CONDITION: id_derivative = { 0,1... (NbVariable-1) }"<<endl<<endl;
    }
    else
    {
        if(ideriv==0)
        {
            value=this->operator()(x);
        }
        else
        {
            for(unsigned int i=0; i<N; i++)
            {
                if (i==ideriv)
                {
                    value*=(Real) powers[i];//derivate
                    value*=(Real) pow(x[i],(powers[i]-1));
                }
                else
                {
                    value*=(Real) pow(x[i],powers[i]);
                }
            }
        }
    }
    return value;
}
////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N> Monomial_LD<Real,N>::d(const unsigned int & ideriv) const
{
    Monomial_LD<Real,N> r(*this);
    if (ideriv >= N)
    {
        cout<<"WARNING : "<<ideriv<<"-th derivative couldn't take place for the monomial of:"<<N<<"-variables"<<endl
            <<r<<endl
            <<"CONDITION: id_derivative = { 0,1... (NbVariable-1) }"<<endl<<endl;
    }
    else
    {
        r.coef*=(Real) r.powers[ideriv];
        if (powers[ideriv] != 0)
        {
            (r.powers[ideriv])--;
        }
    }
    return r;
}
////////////////////////////////
template<typename Real, unsigned int N>
void Monomial_LD<Real,N>::writeToStream(ostream & stream) const
{
    stream<<coef<<"*"<<variables[0]<<"^"<<powers[0];
    for(unsigned int i=1; i<N; i++) stream<<"."<<variables[i]<<"^"<<powers[i];
}
////////////////////////////////
template<typename Real, unsigned int N>
void Monomial_LD<Real,N>::readFromStream(std::istream & stream)
{

    Real t_coef; int t_power; unsigned int counter=0;

    if (stream >> t_coef ) coef=t_coef;

    while ( (counter < N) && (stream >> t_power) )
    {
        powers[counter]=t_power;
        counter++;
    }

    if( stream.rdstate() & std::ios_base::eofbit ) { stream.clear(); }
}

template< typename FReal, unsigned int FN > //For comutativity of operator *: Monomial_LD*Real || Real*Monomial_LD.
Monomial_LD< FReal, FN > & operator*(const FReal & alpha,Monomial_LD< FReal, FN > & r)
{
    r *= alpha;
    return r;
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>::Polynomial_LD()
{
    Monomial_LD<Real,N> monomialnull;
    listofTerms.push_back(monomialnull);
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>::Polynomial_LD(const Polynomial_LD<Real,N> & a)
{
    listofTerms=a.listofTerms;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>::Polynomial_LD(const Monomial_LD<Real,N> & a)
{
    listofTerms.push_back(a);
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>::Polynomial_LD(const unsigned int & nbofTerm,...)
{
    va_list vl;
    va_start(vl,nbofTerm);
    for (unsigned int iterm=0; iterm<nbofTerm; iterm++)
    {
        Monomial_LD<Real,N> mi;
        vector<int> powermonomiali(N,0);

        Real coefi=va_arg(vl,Real);
        for(unsigned int jvar=0; jvar<N; jvar++)
        {
            powermonomiali[jvar]=va_arg(vl,int);
            //mi.powers[jvar]=va_arg(vl,int);
        }
        mi.SetCoef(coefi); mi.SetPower(powermonomiali);
        listofTerms.push_back(mi);
    }
    va_end(vl);
}
////////////////////////////////
template<typename Real, unsigned int N>
int Polynomial_LD<Real,N>::degree()
{
    int deg=0;
    for(MonomialIterator it=listofTerms.begin(); it != listofTerms.end(); ++it)
    {
        deg = ( (*it).degree() > deg ) ? (*it).degree() : deg;
    }
    return deg;
}
////////////////////////////////
template<typename Real, unsigned int N>
bool Polynomial_LD<Real,N>::operator ==(const Polynomial_LD<Real,N> & b) const
{
    bool result=true;
    if ( this->listofTerms.size() != b.listofTerms.size() )
        result=false;
    else
    {
        result = ( this->listofTerms == b.listofTerms ) ? result : false;
    }
    return result;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator*=(const Real & alpha)
{
    for(MonomialIterator it=listofTerms.begin(); it != listofTerms.end(); ++it)
    {
        (*it)*=alpha;
    }
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator/=(const Real & alpha)
{
    for(MonomialIterator it=listofTerms.begin(); it != listofTerms.end(); ++it)
    {
        //*it->operator/=(alpha);
        (*it)/=alpha;
    }
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator+=(const Monomial_LD<Real,N> & b)
{
    bool added=false;
    for(MonomialIterator ita=listofTerms.begin(); ita != listofTerms.end(); ++ita)
    {
        if ( (*ita).isSamePowers(b) )
        {
            (*ita)+=b;
            added=true;
            break;
        }
    }
    if (!added) listofTerms.push_back(b);
    return *this;
}
////////////////////////////////
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator+=(const Polynomial_LD<Real,N> & b)
{
    for(MonomialConstIterator itb=b.listofTerms.begin(); itb != b.listofTerms.end(); ++itb)
    {
        bool added=false;
        for(MonomialIterator ita=listofTerms.begin(); ita != listofTerms.end(); ++ita)
        {
            if ( (*ita).isSamePowers(*itb) )
            {
                (*ita)+=(*itb);
                added=true;
                break;
            }
        }
        if (!added) listofTerms.push_back((*itb));
    }
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator-=(const Polynomial_LD<Real,N> & b)
{
    for(MonomialConstIterator itb=b.listofTerms.begin(); itb != b.listofTerms.end(); ++itb)
    {
        bool added=false;
        for(MonomialIterator ita=listofTerms.begin(); ita != listofTerms.end(); ++ita)
        {
            if ( (*ita).isSamePowers(*itb) )
            {
                (*ita)-=(*itb);
                added=true;
                break;
            }
        }
        if (!added) listofTerms.push_back(-(*itb));
    }
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator*=(const Polynomial_LD<Real,N> & b)
{
    MonomialIterator ita=listofTerms.begin();
    while(ita != listofTerms.end())
    {
        for(MonomialConstIterator itb=b.listofTerms.begin(); itb != b.listofTerms.end(); ++itb)
        {
            Monomial_LD<Real,N> multipSimple=(*ita)*(*itb);
            listofTerms.insert(ita,multipSimple);
        }
        ita=listofTerms.erase(ita);
        //++ita;
    }
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N> Polynomial_LD<Real,N>::operator-() const
{
    Polynomial_LD<Real,N> r(*this);
    for(MonomialIterator it=r.listofTerms.begin(); it != r.listofTerms.end(); ++it)
    {
        (*it).coef*=(Real) -1.;
    }
    return r;
}
////////////////////////////////

template<typename Real, unsigned int N>
Real Polynomial_LD<Real,N>::operator()(const sofa::helper::vector<Real> & x) const
{
    Real result=(Real) 0.;
    for(MonomialConstIterator it=listofTerms.begin(); it != listofTerms.end(); ++it)
    {
        result += (*it).operator()(x);
    }
    return result;
}

////////////////////////////////
template<typename Real, unsigned int N>
Real Polynomial_LD<Real,N>::operator()(const RNpoint & x) const
{
    Real result=(Real) 0.;
    for(MonomialConstIterator it=listofTerms.begin(); it != listofTerms.end(); ++it)
    {
        result += (*it).operator()(x);
    }
    return result;
}
////////////////////////////////
template<typename Real, unsigned int N>
Real Polynomial_LD<Real,N>::operator()(const sofa::helper::vector<Real> & x,unsigned int iderive) const
{
    Real result=(Real) 0.;
    if (iderive >= N)
    {
        cout<<"WARNING : "<<iderive<<"-th derivative couldn't take place for the polynomial of:"<<N<<"-variables"<<endl
            <<(*this)<<endl
            <<"CONDITION: id_derivative = { 0,1... (NbVariable-1) }"<<endl<<endl;
    }
    else
    {
        if(iderive==0)
        {
            result=this->operator()(x);
        }
        else
        {
            for(MonomialConstIterator it=listofTerms.begin(); it != listofTerms.end(); ++it)
            {
                result += (*it).operator()(x,iderive);
            }
        }

    }
    return result;
}

////////////////////////////////
template<typename Real, unsigned int N>
Real Polynomial_LD<Real,N>::operator()(const RNpoint & x,unsigned int iderive) const
{
    Real result=(Real) 0.;
    if (iderive >= N)
    {
        cout<<"WARNING : "<<iderive<<"-th derivative couldn't take place for the polynomial of:"<<N<<"-variables"<<endl
            <<(*this)<<endl
            <<"CONDITION: id_derivative = { 0,1... (NbVariable-1) }"<<endl<<endl;
    }
    else
    {
        if(iderive==0)
        {
            result=this->operator()(x);
        }
        else
        {
            for(MonomialConstIterator it=listofTerms.begin(); it != listofTerms.end(); ++it)
            {
                result += (*it).operator()(x,iderive);
            }
        }

    }
    return result;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N> Polynomial_LD<Real,N>::d(const unsigned int & iderive) const
{
    Polynomial_LD<Real,N> result(*this);
    if (iderive >=N)
    {
        cout<<"WARNING : "<<iderive<<"-th derivative couldn't take place for the polynomial of:"<<"-variables"<<endl
            <<result<<endl
            <<"CONDITION: id_derivative = { 0,1... (NbVariable-1) }"<<endl<<endl;
    }
    else
    {
        for(MonomialIterator it=result.listofTerms.begin(); it != result.listofTerms.end(); ++it)
        {
            (*it).coef*=(Real) (*it).powers[iderive];
            if ((*it).powers[iderive] != 0)
            {
                ((*it).powers[iderive])--;
            }
        }
    }
    return result;
}
////////////////////////////////
////////////////////////////////
template<typename Real, unsigned int N>
void Polynomial_LD<Real,N>::writeToStream(std::ostream & stream) const
{
    MonomialConstIterator it=listofTerms.begin();
    stream<< *it; ++it;
    while(it != listofTerms.end() )
    {
        stream << "  +  "<<*it;
        ++it;
    }
}
////////////////////////////////
template<typename Real, unsigned int N>
void Polynomial_LD<Real,N>::readFromStream(std::istream & stream)
{
    listofTerms.clear();
    Monomial_LD<Real,N> tempo;

    while (stream >> tempo)
    {
        listofTerms.push_back(tempo);
    }

    if( stream.rdstate() & std::ios_base::eofbit ) { stream.clear(); }
}
////////////////////////////////
template< typename FReal, unsigned int FN >
Polynomial_LD< FReal, FN > & operator*(const FReal & alpha, Polynomial_LD< FReal, FN> & r)
{
    r *= alpha;
    return r;
}
////////////////////////////////
template< typename FReal, unsigned int FN >
Polynomial_LD< FReal, FN > & operator*(const Monomial_LD< FReal, FN >   & a, Polynomial_LD< FReal, FN> & r)
{
    r *= a;
    return r;
}


} // namespace helper

} // namespace sofa

#endif

