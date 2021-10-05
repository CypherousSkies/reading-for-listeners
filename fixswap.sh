sudo swapoff -a
# size = bs*count Bytes, currently 10GiB
sudo dd if=/dev/zero of=/swapfile bs=1024 count=10485760 status=progress
sudo chmod 0600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show
