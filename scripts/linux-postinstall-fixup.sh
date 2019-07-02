#!/bin/bash

usage() {
    echo "Usage: linux-postinstall-fixup.sh <install-dir> <qt-dir>"
}

if [ "$#" -ge 2 ]; then
    INSTALL_DIR="$(cd $1 && pwd)"
    
    QT_DIR=""
	if [ -d "$2" ]; then
		QT_DIR="$(cd $2 && pwd)"
	fi
else
    usage; exit 1
fi

echo "Fixing up libs..."

# Why are these folders installed in plugins?
rm -rf "$INSTALL_DIR/plugins/iconengines"
rm -rf "$INSTALL_DIR/plugins/imageformats"
rm -rf "$INSTALL_DIR/plugins/platforms"
rm -rf "$INSTALL_DIR/plugins/styles"


get-lib-deps() {
	local base_dir="$1"
	local libs="$(find "$base_dir" -type f -name "*.so*" -path "$base_dir/lib/*" ; find "$base_dir" -type f -name "*.so*" -path "$base_dir/bin/*")"
		
	#echo " ------- libs --------"
	#echo "$libs"
	#echo "----------------------"

	ldd $libs | grep " => [^ \.].* " | grep -v "$base_dir" | cut -c2- | sed -e 's/\(.*\) => .*/\1/g' | sort | uniq
}


echo "  Searching missing deps in $INSTALL_DIR"

lib_deps="$(get-lib-deps "$INSTALL_DIR")"

# Copy Qt libs (and their deps)
if [ -n "$QT_DIR" ]; then
	qt_deps="$(echo $lib_deps | tr " " "\n" | grep "libQt")"

	qt_deps_to_copy="$qt_deps"
	for qtlib in $qt_deps; do
		qt_deps_to_copy="$qt_deps_to_copy $(ldd "$QT_DIR/lib/$qtlib" | grep "$QT_DIR" | cut -c2- | sed -e 's/\(.*\) => .*/\1/g')"
	done
	qt_deps_to_copy="$(echo $qt_deps_to_copy | tr " " "\n" | sort | uniq)"

	for qtlib in $qt_deps_to_copy; do
		echo "    $qtlib"
		cp -Rf "$QT_DIR/lib/$qtlib"* "$INSTALL_DIR/lib"
	done
fi

# Copy libPNG
if echo "$lib_deps" | grep -q "libpng12"; then
    echo "    libpng12.so"
    cp -Rf "/lib/x86_64-linux-gnu/libpng12.so"* "$INSTALL_DIR/lib"
fi

echo "Done."
exit 0

######################################

# TODO
# Check if the system deps are in default system packages
# and copy them if not

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
