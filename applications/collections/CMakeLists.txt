cmake_minimum_required(VERSION 3.1)

sofa_add_plugin(SofaComponentBase SofaComponentBase ON)
sofa_add_plugin(SofaComponentCommon SofaComponentCommon ON)
sofa_add_plugin(SofaComponentGeneral SofaComponentGeneral ON)
sofa_add_plugin(SofaComponentMisc SofaComponentMisc ON)
sofa_add_plugin(SofaComponentAdvanced SofaComponentAdvanced ON)

sofa_add_plugin(SofaComponentAll SofaComponentAll ON) # Depends on all SofaComponent*
