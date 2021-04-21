#!/bin/bash
echo "installing nextEPC..."
sudo apt-get remove -y --purge man-db
sudo apt-get update
sudo apt-get -y install mongodb
sudo systemctl start mongodb
sudo apt-get -y install autoconf libtool gcc pkg-config \
         git flex bison libsctp-dev libgnutls28-dev libgcrypt-dev \
         libssl-dev libidn11-dev libmongoc-dev libbson-dev libyaml-dev
curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
sudo apt-get -y install nodejs

cd /opt
sudo git clone https://github.com/nextepc/nextepc
cd nextepc
sudo autoreconf -iv
sudo ./configure --prefix=`pwd`/install
sudo make -j `nproc`
sudo make install

cd /opt/nextepc/webui
{
sudo npm install &&


sudo cat << EOF > /etc/systemd/network/98-nextepc.netdev
[NetDev]
Name=pgwtun
Kind=tun
EOF

    sudo systemctl restart systemd-networkd &&
    sudo ip addr add 192.168.0.1/24 dev pgwtun &&
    sudo ip link set up dev pgwtun &&
    sudo iptables -t nat -A POSTROUTING -o `cat /var/emulab/boot/controlif` -j MASQUERADE &&
    sudo cp /local/repository/etc/NextEPC/nextepc.conf /opt/nextepc/install/etc/nextepc/nextepc.conf
} || {
sudo cat << EOF > /etc/systemd/network/98-nextepc.netdev
[NetDev]
Name=pgwtun
Kind=tun
EOF

sudo systemctl restart systemd-networkd
sudo ip addr add 192.168.0.1/24 dev pgwtun
sudo ip link set up dev pgwtun
sudo iptables -t nat -A POSTROUTING -o `cat /var/emulab/boot/controlif` -j MASQUERADE
sudo cp /local/repository/etc/NextEPC/nextepc.conf /opt/nextepc/install/etc/nextepc/nextepc.conf
}
