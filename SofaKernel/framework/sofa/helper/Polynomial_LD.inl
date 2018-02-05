/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_POLYNOMIAL_LD_INL
#define SOFA_HELPER_POLYNOMIAL_LD_INL

#include <sofa/helper/Polynomial_LD.h>
#include <sstream>
#include <iterator>


namespace sofa
{
namespace helper
{

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
        std::ostringstream oss; oss << 't' << i ;
        variables[i]=oss.str();
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
        dmsg_warning("Monomial_LD") <<(*this)<<" + "<<b <<"   Not permissed with different powers" ;
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
        dmsg_warning("Monomial_LD") <<(*this)<<" - "<<b <<"   Not permissed with different powers" ;
    }
    return *this;
}

////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N> & Monomial_LD<Real,N>::operator*=(const Monomial_LD<Real,N> & b)
{
    ///////////////////////////////////////////////////////
    coef*=b.coef;
    for(unsigned int i=0; i<N; i++)
        powers[i] += b.powers[i];

    return *this;
}

////////////////////////////////
template<typename Real, unsigned int N>
Real Monomial_LD<Real,N>::operator()(const RNpoint & x) const
{
    dmsg_warning_when(x.size()!= N, "Monomial_LD") << "Value assigned to the monome has not the good number of variable." ;
    Real value= coef;
    for( unsigned int i=0; i<N; i++)
    {
        value*=(Real) pow(x[i],powers[i]);
    }
    return value;
}

////////////////////////////////
template<typename Real, unsigned int N>
Real Monomial_LD<Real,N>::operator()(const RNpoint & x,unsigned int idvar) const
{
    dmsg_warning_when(x.size()!= N, "Monomial_LD") <<"Value assigned to the monome has not the good number of variable" ;

    Real value= coef;

    if (idvar >= N)
    {
        dmsg_warning("Monomial_LD") <<idvar<<"-th partial derivative couldn't take place for the monomial of:"<<N<<"-variables"<<msgendl
                                    <<(*this)<< msgendl
                                    <<"CONDITION: id_variable = { 0,1... (NbVariable-1) }";
    }
    else
    {
        for(unsigned int i=0; i<N; i++)
        {
            if (i!=idvar)
            {
                value*=(Real) pow(x[i],powers[i]);
            }
            else if(powers[i]>0)
            {
                value*=(Real) powers[i];//derivate
                value*=(Real) pow(x[i],(powers[i]-1));
            }
            else
            {
                value *= (Real)0.;
            }
        }

    }
    return value;
}
////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N> Monomial_LD<Real,N>::d(const unsigned int & idvar) const
{
    Monomial_LD<Real,N> r(*this);
    if (idvar >= N)
    {
        msg_warning("Monomial_LD") << idvar<<"-th derivative couldn't take place for the monomial of:"<<N<<"-variables"<<msgendl
                                   <<r<<msgendl
                                  <<"CONDITION: id_derivative = { 0,1... (NbVariable-1) }" ;
    }
    else
    {
        r.coef*=(Real) r.powers[idvar];
        if (powers[idvar] != 0)
        {
            (r.powers[idvar])--;
        }
    }
    return r;
}
////////////////////////////////
template<typename Real, unsigned int N>
void Monomial_LD<Real,N>::writeToStream(std::ostream & stream) const
{
    stream<<coef<<"*"<<variables[0]<<"^"<<powers[0];
    for(unsigned int i=1; i<N; i++) stream<<"*"<<variables[i]<<"^"<<powers[i];
}
////////////////////////////////
template<typename Real, unsigned int N>
void Monomial_LD<Real,N>::readFromStream(std::istream & stream)
{
    Real t_coef; int t_power;

    if (stream >> t_coef ) coef=t_coef;

    for(unsigned int i=0; i<N; ++i)
    {
        if (stream >> t_power) powers[i]=t_power;
    }
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
    nbOfMonomial=0;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>::Polynomial_LD(const Polynomial_LD<Real,N> & a)
{
    listOfMonoMial=a.listOfMonoMial;
    nbOfMonomial=a.nbOfMonomial;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>::Polynomial_LD(const Monomial_LD<Real,N> & a)
{
    listOfMonoMial.push_back(a);
    nbOfMonomial=1;
}
////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////
template<typename Real, unsigned int N>
Monomial_LD<Real,N>::Monomial_LD(Real m_coef, ...)
{
    coef=m_coef;
    va_list vl;
    va_start(vl,m_coef);
    for(unsigned int i=0; i<N; i++)
    {
        powers[i]=va_arg(vl,int);
        std::ostringstream oss; oss << 'l' << i ;
        variables[i]=oss.str();
    }
    va_end(vl);
}

template<typename Real, unsigned int N>
Polynomial_LD<Real,N>::Polynomial_LD(const unsigned  int nbofTerm,...)
{
    nbOfMonomial=nbofTerm;

    va_list vl;
    va_start(vl,nbofTerm);

    for (unsigned int iterm=0; iterm<nbofTerm; iterm++)
    {
        Monomial_LD<Real,N> mi;
        double coefindouble=va_arg(vl, double );
        Real coefi=(Real)coefindouble;
        mi.SetCoef(coefi);
        for(unsigned int jvar=0; jvar<N; jvar++)
        {
            int m_power=va_arg(vl,int); mi.SetPower(jvar,m_power);
        }

        listOfMonoMial.push_back(mi);
    }
    va_end(vl);
    this->sort();
}
////////////////////////////////
template<typename Real, unsigned int N>
void Polynomial_LD<Real,N>::Set(const unsigned int nbofTerm,...)
{
    nbOfMonomial= nbofTerm;
    if(nbOfMonomial!=nbofTerm)
    {
        nbOfMonomial=nbofTerm  ;  listOfMonoMial.resize(nbOfMonomial);
    }

    va_list vl;
    va_start(vl,nbofTerm);

    for (unsigned int iterm=0; iterm<nbofTerm; iterm++)
    {

        Monomial_LD<Real,N> mi;
        double coefindouble=va_arg(vl, double );
        Real coefi=(Real)coefindouble;
        mi.SetCoef(coefi); mi.SetCoef(coefi);

        for(unsigned int jvar=0; jvar<N; jvar++)
        {
            int m_power=va_arg(vl,int); mi.SetPower(jvar,m_power);
        }

        listOfMonoMial[iterm]=mi;
    }
    va_end(vl);
    this->sort();
}
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////




////////////////////////////////
template<typename Real, unsigned int N>
int Polynomial_LD<Real,N>::degree()
{
    int deg=0;
    for(MonomialIterator it=listOfMonoMial.begin(); it != listOfMonoMial.end(); ++it)
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
    if ( this->listOfMonoMial.size() != b.listOfMonoMial.size() )
        result=false;
    else
    {
        result = ( this->listOfMonoMial == b.listOfMonoMial ) ? result : false;
    }
    return result;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator*=(const Real & alpha)
{
    for(MonomialIterator it=listOfMonoMial.begin(); it != listOfMonoMial.end(); ++it)
    {
        (*it)*=alpha;
    }
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator/=(const Real & alpha)
{
    for(MonomialIterator it=listOfMonoMial.begin(); it != listOfMonoMial.end(); ++it)
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
    for(MonomialIterator ita=listOfMonoMial.begin(); ita != listOfMonoMial.end(); ++ita)
    {
        if ( (*ita).isSamePowers(b) )
        {
            (*ita)+=b;
            added=true;
            break;
        }
    }
    if (!added)
    {
        listOfMonoMial.push_back(b);
        nbOfMonomial++;
    }
    sort();
    return *this;
}
////////////////////////////////
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator+=(const Polynomial_LD<Real,N> & b)
{
    for(MonomialConstIterator itb=b.listOfMonoMial.begin(); itb != b.listOfMonoMial.end(); ++itb)
    {
        bool added=false;
        for(MonomialIterator ita=listOfMonoMial.begin(); ita != listOfMonoMial.end(); ++ita)
        {
            if ( (*ita).isSamePowers(*itb) )
            {
                (*ita)+=(*itb);
                added=true;
                break;
            }
        }
        if (!added)
        {
            listOfMonoMial.push_back((*itb));
            nbOfMonomial++;
        }
    }
    sort();
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator-=(const Polynomial_LD<Real,N> & b)
{
    for(MonomialConstIterator itb=b.listOfMonoMial.begin(); itb != b.listOfMonoMial.end(); ++itb)
    {
        bool added=false;
        for(MonomialIterator ita=listOfMonoMial.begin(); ita != listOfMonoMial.end(); ++ita)
        {
            if ( (*ita).isSamePowers(*itb) )
            {
                (*ita)-=(*itb);
                added=true;
                break;
            }
        }
        if (!added)
        {
            listOfMonoMial.push_back(-(*itb));
            nbOfMonomial++;
        }
    }
    sort();
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N>  & Polynomial_LD<Real,N>::operator*=(const Polynomial_LD<Real,N> & b)
{
    Polynomial_LD<Real,N> old=(*this);
    listOfMonoMial.resize(listOfMonoMial.size()*(b.listOfMonoMial.size()));

    unsigned int m_counter=0;
    for(unsigned int ita=0; ita<old.listOfMonoMial.size(); ita++)
    {
        for(unsigned int itb=0; itb<b.listOfMonoMial.size(); itb++)
        {
            listOfMonoMial[m_counter]=old.listOfMonoMial[ita]*b.listOfMonoMial[itb];
            m_counter++;
        }
    }

    this->sort();
    return *this;
}
////////////////////////////////
template<typename Real, unsigned int N>
Polynomial_LD<Real,N> Polynomial_LD<Real,N>::operator-() const
{
    Polynomial_LD<Real,N> r(*this);
    for(MonomialIterator it=r.listOfMonoMial.begin(); it != r.listOfMonoMial.end(); ++it)
    {
        (*it).coef*=(Real) -1.;
    }
    r.sort();
    return r;
}
////////////////////////////////
template<typename Real, unsigned int N>
Real Polynomial_LD<Real,N>::operator()(const RNpoint & x) const
{
    Real result=(Real) 0.;
    for(MonomialConstIterator it=listOfMonoMial.begin(); it != listOfMonoMial.end(); ++it)
    {
        result += (*it).operator()(x);
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
        dmsg_warning("Polynomial_LD") <<iderive<<"-th derivative couldn't take place for the polynomial of:"<<N<<"-variables"<<msgendl
            <<(*this)<<msgendl
            <<"CONDITION: id_derivative = { 0,1... (NbVariable-1) }" ;
    }
    else
    {
        for(MonomialConstIterator it=listOfMonoMial.begin(); it != listOfMonoMial.end(); ++it)
        {
            result += it->operator()(x,iderive);
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
        dmsg_warning("Polynomial_LD") <<iderive<<"-th derivative couldn't take place for the polynomial of:"<<"-variables" << msgendl
            <<result<<msgendl
            <<"CONDITION: id_derivative = { 0,1... (NbVariable-1) }" ;
    }
    else
    {
        for(MonomialIterator it=result.listOfMonoMial.begin(); it != result.listOfMonoMial.end(); ++it)
        {
            (*it).coef*=(Real) (*it).powers[iderive];
            if ((*it).powers[iderive] != 0)
            {
                ((*it).powers[iderive])--;
            }
        }
    }
    result.sort();
    return result;
}
////////////////////////////////
////////////////////////////////
template<typename Real, unsigned int N>
void Polynomial_LD<Real,N>::setnbOfMonomial(int m_nbofmonomial)
{
    listOfMonoMial.clear();
    nbOfMonomial=m_nbofmonomial;
    Monomial_LD<Real,N> monomialNULL;
    for(unsigned int i=0; i<nbOfMonomial; i++)
    {
        listOfMonoMial.push_back(monomialNULL);
    }
}
////////////////////////////////
template<typename Real, unsigned int N>
void Polynomial_LD<Real,N>::writeToStream(std::ostream & stream) const
{
    MonomialConstIterator it=listOfMonoMial.begin();
    stream<< *it; ++it;
    while(it != listOfMonoMial.end() )
    {
        stream << "  +  "<<*it;
        ++it;
    }
}
////////////////////////////////
template<typename Real, unsigned int N>
void Polynomial_LD<Real,N>::readFromStream(std::istream & stream)
{
    int nbofmonomialtempo;

    if (stream >> nbofmonomialtempo) nbOfMonomial=nbofmonomialtempo;

    listOfMonoMial.resize(nbOfMonomial);
    MonomialIterator it=listOfMonoMial.begin();
    for(unsigned int monomialcounter=0; monomialcounter<nbOfMonomial; ++monomialcounter)
    {
        Monomial_LD<Real,N> tempo;
        if (stream >> tempo) (*it)=tempo;
        ++it;
    }
    sort();
}
////////////////////////////////
template<typename Real, unsigned int N>
std::string  Polynomial_LD<Real,N>::getString() const
{
    std::ostringstream m_outstream;
    m_outstream<<(*this);
    return m_outstream.str();
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
    r.sort();
    return r;
}
////////////////////////////////
template<typename Real, unsigned int N>
void Polynomial_LD<Real,N>::exchangeMonomial(unsigned int ithMono,unsigned  int jthMono)
{
    Monomial_LD<Real,N> tempo;
    tempo=listOfMonoMial[ithMono];
    listOfMonoMial[ithMono]=listOfMonoMial[jthMono];
    listOfMonoMial[jthMono]=tempo;
}
////////////////////////////////
template<typename Real, unsigned int N>
void Polynomial_LD<Real,N>::sortByVar(unsigned int idVar)
{
    if(idVar==0) //First variable in Rd
    {
        for(unsigned int ith=0; ith < listOfMonoMial.size()-1; ith++)
            for(unsigned int jth=ith+1; jth < listOfMonoMial.size(); jth++)
            {
                if (listOfMonoMial[ith].powers[idVar] < listOfMonoMial[jth].powers[idVar])
                {
                    exchangeMonomial(ith,jth);
                }
            }
    }
    else
    {
        for(unsigned int ith=0; ith<listOfMonoMial.size()-1; ith++)
            for(unsigned int jth=ith+1; jth<listOfMonoMial.size(); jth++)
            {
                if (   (listOfMonoMial[ith].powers[idVar] < listOfMonoMial[jth].powers[idVar])
                        && (listOfMonoMial[ith].powers[idVar-1] == listOfMonoMial[ith].powers[idVar-1])	)
                {
                    exchangeMonomial(ith,jth);
                }
            }
    }
}
////////////////////////////////
template<typename Real, unsigned int N>
void Polynomial_LD<Real,N>::sort()
{
    //fusion all monomials which differ only the coef
    for(MonomialIterator ita=listOfMonoMial.begin(); ita != listOfMonoMial.end() ; ++ita)
    {
        MonomialIterator itb=ita; itb++;
        if (itb==listOfMonoMial.end()) break;

        while(itb!=listOfMonoMial.end())
        {
            if (ita->isSamePowers((*itb)))
            {
                ita->coef += (*itb).coef;
                itb=listOfMonoMial.erase(itb); itb--;
            }
            ++itb;
        }
    }


    //Eliminate all term null monomial
    for(MonomialIterator ita=listOfMonoMial.begin(); ita != listOfMonoMial.end() ; ++ita)
    {
        if ((*ita).coef == 0)
        {
            ita=listOfMonoMial.erase(ita); ita--;
        }
    }

    //Sorting monomial by variable degree
    for(unsigned int ithVar=0; ithVar<N; ithVar++)
    {
        sortByVar(ithVar);
    }
}

} // namespace helper

} // namespace sofa

#endif

