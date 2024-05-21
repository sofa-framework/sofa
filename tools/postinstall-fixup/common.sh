#!/bin/bash

function clean_default_plugins()
{
  # Keep plugin_list as short as possible
  echo "" > "$1/plugin_list.conf"
  disabled_plugins='plugins_ignored_by_default'
  for plugin in \
          Geomagic                  \
          image                     \
          CImgPlugin                \
          PluginExample             \
          SofaCUDA                  \
          CGALPlugin                \
      ; do
      disabled_plugins=$disabled_plugins'\|'$plugin
  done
  grep -v $disabled_plugins "$1/plugin_list.conf.default" >> "$1/plugin_list.conf"
}