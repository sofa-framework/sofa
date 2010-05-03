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
