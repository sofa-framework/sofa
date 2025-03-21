#! /usr/bin/awk -f
# This script removes core::componentmodel namespace
# The file with the relevant parts removed is output to the standard output.

# init
BEGIN {
    skipNext=0;
	skipNextSpaceLine=0
	skipCur=0;
}

# parse "namespace componentmodel"
#		"{
skipNext==1 && $1=="{" {
	skipNextSpaceLine=1;
	next;
}

# parse newline after namespace componentmodel declaration
skipNextSpaceLine==1 && $0=="" {
	skipNext=0;
	next;
}

# parse "namespace componentmodel {"
$1=="namespace" && $2=="componentmodel" && $3=="{" {
	skipNextSpaceLine=1;
	next;
}

# parse "namespace componentmodel"
#		"{
$1=="namespace" && $2=="componentmodel" {
	skipNext=1;
    next;
}

# parse "} // namespace componentmodel"
$1=="}" && $2=="//" && $3=="namespace" && $4=="componentmodel" {
	skipNextSpaceLine=1;
	next;
}

#parse "using namespace sofa::core::componentmodel;"
$1=="using" && $2=="namespace" && $3=="sofa::core::componentmodel;" {
	skipNextSpaceLine=1;
	next;
}

#parse "using namespace core::componentmodel;"
$1=="using" && $2=="namespace" && $3=="core::componentmodel;" {
	skipNextSpaceLine=1;
	next;
}

skipNext==1 {
	skipNext=0; 
}

skipNextSpaceLine==1 {
	skipNextSpaceLine=0;
}


{
	# remove _COMPONENTMODEL_ in define
	gsub("_COMPONENTMODEL_", "_")
	# remove componentmodel in include
	gsub("componentmodel/", "")
	#remove componentmodel  in namespace
	gsub("componentmodel::", "")
	
	print $0;
}

