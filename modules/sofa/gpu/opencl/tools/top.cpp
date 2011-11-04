/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "top.h"
#include <sys/time.h>
#include <unistd.h>
#include <iostream>

/// fonctions pour le rendu du temps du temps
#define K_MAX 100
unsigned long long t[K_MAX] = {0};	//chaine regroupant toutes les durées
int lines[K_MAX] = {0};			//numéro des lignes où ont été effectués les top;
int k=0;				//nombrte de durées receuillie

void top(int line)
{
    if(k>=K_MAX)return;
    lines[k]=line;
    struct timeval tv;
    gettimeofday(&tv,0);
    t[k++]=((unsigned long long)tv.tv_sec)*1000000+((unsigned long long)tv.tv_usec);
}


void topLog()
{
    int i;
    std::cout << "\n========\nTIME_LOG\n  ====  \n\n";
    for(i=1; i<k; i++)std::cout << "top(" << lines[i-1] << "-" << lines[i] <<") = (µs)" << t[i]-t[i-1] << "\n";
    std::cout << "\n  ====  \nTIME_LOG\n========\n\n";
    k=0;
}
