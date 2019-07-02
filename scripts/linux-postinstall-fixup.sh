#!/bin/bash

usage() {
    echo "Usage: linux-postinstall-fixup.sh <install-dir> <qt-dir>"
}

if [ "$#" -ge 1 ]; then
    INSTALL_DIR="$(cd $1 && pwd)"
    QT_DIR="$(cd $2 && pwd)"
else
    usage; exit 1
fi

echo "Fixing up libs..."

get-lib-deps() {
	local base_dir="$1"
	local libs="$(find "$base_dir" -type f -name "*.so*" -path "$base_dir/lib/*" ; find "$base_dir" -type f -name "*.so*" -path "$base_dir/bin/*")"
		
	#echo " ------- libs --------"
	#echo "$libs"
	#echo "----------------------"

	ldd $libs | grep " => [^ \.].* " | grep -v "$base_dir" | cut -c2- | sed -e 's/\(.*\) => .*/\1/g' | sort | uniq
}

lib_deps="$(get-lib-deps "$INSTALL_DIR")"

#echo " ------- lib deps --------"
#echo "$lib_deps" | tr " " "\n"
#echo "--------------------------"

qt_deps="$(echo $lib_deps | tr " " "\n" | grep "libQt")"

#echo " -------- Qt deps --------"
#echo "$qt_deps" | tr " " "\n"
#echo "--------------------------"

qt_deps_to_copy="$qt_deps"
for qtlib in $qt_deps; do
	qt_deps_to_copy="$qt_deps_to_copy $(ldd "$QT_DIR/lib/$qtlib" | grep "$QT_DIR" | cut -c2- | sed -e 's/\(.*\) => .*/\1/g')"
done
qt_deps_to_copy="$(echo $qt_deps_to_copy | tr " " "\n" | sort | uniq)"

echo " ---- Qt deps to copy ----"
echo "$qt_deps_to_copy" | tr " " "\n"
echo "--------------------------"

for qtlib in $qt_deps_to_copy; do
	cp -Rf "$QT_DIR/lib/$qtlib"* "$INSTALL_DIR/lib"
done

exit 0

######################################

# TODO
# Check if some system deps are not from default system packages

# refresh libs to integrate new Qt libs
lib_deps="$(get-lib-deps "$INSTALL_DIR")"

echo " ------- lib_deps --------"
echo "$lib_deps"
echo "--------------------------"

rm -f /tmp/linux_postinstall_fixup_manifest
version="$(lsb_release -a 2>/dev/null | grep "Description" | sed -e 's/.* \([0-9][0-9]\.[0-9][0-9]\.[0-9]*\) .*/\1/g')"
codename="$(lsb_release -a 2>/dev/null | grep "Codename" | sed -e 's/.*\t\(.*\)/\1/g')"
manifest="/tmp/ubuntu-$version-desktop-amd64.manifest"
rm -f "$manifest"
wget --quiet "http://releases.ubuntu.com/$codename/ubuntu-$version-desktop-amd64.manifest" -O "$manifest"

for lib in $lib_deps; do
	package="$(dpkg -S $lib | sed -e s/:.*//g)"
#	if 
#		...
#	fi
done
