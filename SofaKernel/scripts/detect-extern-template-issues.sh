#! /bin/bash -e

# This script tries to detect issues with extern templates in Sofa components.
# Namely, it looks for extern template declarations with no corresponding
# explicit instanciations, and vice versa.  Also, it looks for extern template
# declarations not protected by a !defined(FOO_BAR_CPP) preprocessor thing.
#
#
# It it based on the following assumptions, that are more or less conventions in
# the repository: if Foo is a templated component, it is written in three files:
# Foo.h, Foo.inl and Foo.cpp; the extern template declarations are in Foo.h, and
# the explicit template instanciations are in Foo.cpp.  Also, each declaration
# and each instanciation is on a single line.


usage() {
    echo "Usage: $0 DIRECTORY..."
}

filter-commented-lines() {
    grep '^[[:blank:]]*//' $*
}

filter-extern-template() {
    grep '^[[:blank:]]*extern[[:blank:]]\+template[[:blank:]]\+class' $* | filter-commented-lines -v
}

filter-template-instanciations() {
    grep '^[[:blank:]]*template[[:blank:]]\+class' $* | filter-commented-lines -v
}

delete-spaces() {
    sed 's/[[:blank:]]//g'
}

extract-class-name() {
    sed 's/.*template[[:blank:]]\+class[[:blank:]]\+\([A-Za-z_]*_API[[:blank:]]\+\)\?\([^;]*\);[[:blank:]]*/\2/' | delete-spaces | remove-nested-name-specifiers
}

remove-nested-name-specifiers() {
    sed 's/[a-zA-Z0-9]\+:://g'
}

# member <e> <e_1> <e_2> ... <e_n>
member() {
    local e=$1
    shift
    while [ $# != 0 ]; do
        [ "$1" = "$e" ] && return 0
        shift
    done
    return 1
}

# For each extern template declaration, check there is a corresponding
# template instanciation, and vice-versa.
detect-issues-in-component() {
    local header=$1.h
    local cpp=$1.cpp
    local component=$(echo $header | sed 's:.*/\([^/]*\).h$:\1:')
    local COMPONENT=$(echo $component | tr '[:lower:]' '[:upper:]')
    local extern_declarations=$(filter-extern-template $header | extract-class-name)
    local instanciations=$(filter-template-instanciations $cpp | extract-class-name)

    for i in $instanciations; do
        if ! member $i $extern_declarations; then
            echo "$basename.h:0: warning: missing extern template declaration: $i"
        fi
    done

    for e in $extern_declarations; do
        if ! member $e $instanciations; then
            echo "$basename.cpp:0: warning: missing template instanciation: $e"
        fi
    done

    if ! grep -iq ${COMPONENT}_CPP $header; then
        echo "$header:0: warning: found no references to a ${COMPONENT}_CPP macro"
    fi

    if ! grep -iq ${COMPONENT}_CPP $cpp; then
        echo "$cpp:0: warning: found no references to a ${COMPONENT}_CPP macro"
    fi
}

process-directory() {
    if [[ ! -d "$1" ]]; then
        echo "No such directory: $0 <directory>"
        exit 1
    fi

    # For each file containing  extern templace declarations
    git grep -l '^[[:blank:]]*extern[[:blank:]]\+template' $1 | while read file; do
        local dir=$(dirname $file)
        local basename=$dir/$(basename $file .h)
        detect-issues-in-component $basename
    done
}

if [[ "$#" = 0 ]]; then
    usage
    exit 1
fi

while [ "$#" != 0 ]; do
    process-directory $1
    shift
done
