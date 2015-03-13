#! /bin/bash -e

# This script tries to detect issues with extern templates in Sofa components.
# Namely, it looks for extern template declarations with no corresponding
# explicit instanciations, and vice versa.
#
#
# It it based on the following assumptions, that are more or less conventions in
# the repository: if Foo is a templated component, it is written in three files:
# Foo.h, Foo.inl and Foo.cpp; the extern template declarations are in Foo.h, and
# the explicit template instanciations are in Foo.cpp.


usage() {
    echo "Usage: $0 DIRECTORY..."
}

filter-commented-lines() {
    grep '^[[:blank:]]*//' $*
}

filter-extern-template() {
    grep 'extern[[:blank:]]\+template[[:blank:]]\+class' $* | filter-commented-lines -v
}

filter-included-inl() {
    grep '#[[:blank:]]*include' $* | filter-commented-lines -v | grep '\.inl[">]'
}

filter-template-instanciations() {
    grep 'template[[:blank:]]\+class' $* | filter-commented-lines -v
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
detect-missing-stuff-in-component() {
    local header=$1
    local cpp=$2
    local extern_declarations=$(filter-extern-template $header | extract-class-name)
    local instanciations=$(filter-template-instanciations $cpp | extract-class-name)

    for i in $instanciations; do
        if ! member $i $extern_declarations; then
            echo "$basename: missing extern template declaration: $i"
        fi
    done

    for e in $extern_declarations; do
        if ! member $e $instanciations; then
            echo "$basename: missing template instanciation: $e"
        fi
    done
}

# Compare the number of extern template declarations and template
# instanciations.  (I think it might detect different issues; not sure though.)
detect-missing-stuff-in-component-2() {
    local header=$1
    local cpp=$2
    extern_count=$(filter-extern-template $header | wc -l)
    instanciation_count=$(filter-template-instanciations $cpp | wc -l)
    if [[ $extern_count != $instanciation_count ]]; then
        echo "$basename: found $extern_count extern templates, but $instanciation_count template instanciations"
    fi
}

# process-directory() {
#     if [[ ! -d "$1" ]]; then
#         echo "No such directory: $0 <directory>"
#         exit 1
#     fi
#     find $1 -name '*inl' | while read file; do
#         local dir=$(dirname $file)
#         local basename=$dir/$(basename $file .inl)
#         if [[ -e $basename.h && -e $basename.cpp ]]; then
#             detect-missing-stuff-in-component $basename.h $basename.cpp
#             detect-missing-stuff-in-component-2 $basename.h $basename.cpp
#         fi
#     done
# }

# if [[ "$#" = 0 ]]; then
#     usage
#     exit 1
# fi

# while [ "$#" != 0 ]; do
#     process-directory $1
#     shift
# done

extract-component() {
    sed 's:.*include [<"]\(.*\)\.[^\.]*[>"].*:\1:'
}

extract-component-name() {
    # assume a/b
    sed 's:.*include [<"].*/\([^/]\+\)\.[^\.]*[>"].*:\1:'
}

extract-class() {
    local class=$1
    sed -e "s:.*\($class[[:blank:]]*<[[:blank:]]*[^>]\+[[:blank:]]*>\).*:\1:" | delete-spaces | remove-nested-name-specifiers
}


validate() { 
    local inl=$(filter-included-inl $1 | extract-component)
    for c in $inl; do
        case $1 in
            *$c.cpp)
                # echo ignoring $1
                ;;
            *)
                echo $c
                local ok=nope
                externs=$(filter-extern-template modules/$c.h | extract-class-name)
                grep "$c[[:blank:]]*<" $1 | extract-class $c | sort -u | while read i; do
                    if ! member $i $externs; then
                        ok=yep
                    fi
                done
                if [ $ok = nope ]; then
                    echo "$c included"
                fi
                ;;
        esac
    done
}

validate $1

