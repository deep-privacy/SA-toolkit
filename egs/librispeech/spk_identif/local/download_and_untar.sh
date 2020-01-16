#!/bin/bash

dlpath=./download
rm $dlpath -rf
if ! which wget >/dev/null; then
  echo "$0: wget is not installed."
  exit 1;
fi

url=www.openslr.org/resources/12/raw-metadata.tar.gz

echo "$0: downloading data from $url."

if ! wget -P $dlpath --no-check-certificate $url; then
  echo "$0: error executing wget $url"
  exit 1
fi

if ! tar -C $dlpath -xvzf $dlpath/raw-metadata.tar.gz; then
  echo "$0: error un-tarring archive .tar.gz"
  exit 1
fi

rm $dlpath/raw-metadata.tar.gz
