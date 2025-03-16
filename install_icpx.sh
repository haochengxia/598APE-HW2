#!/bin/bash

#////////////////////////////////////////////////////////////////
# install intel oneapi
#////////////////////////////////////////////////////////////////

pushd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
# add to your apt sources keyring so that archives signed with this key will be trusted.
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
# remove the public key
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
# Configure the APT client to use Intel's repository
sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
# install package
sudo apt install intel-basekit
# activate variables (can be added to ~/.bashrc)
. /opt/intel/oneapi/setvars.sh intel64
popd