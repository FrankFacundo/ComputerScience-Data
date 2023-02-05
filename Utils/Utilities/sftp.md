SFTP:

0. config: https://linuxhint.com/setup-sftp-server-ubuntu/

1. First share hard drice with other users (see below).x

2. change default directory: 
https://community.jaguar-network.com/restreindre-lacces-sftp-dun-utilisateur-chroot/

vim /etc/ssh/sshd_config
nano /etc/ssh/sshd_config

    ## Configuration à ajouter en fin de fichier
    Subsystem       sftp    internal-sftp

    Match Group groupe_restreint
            ChrootDirectory /home/%u
            ForceCommand internal-sftp
            AllowTCPForwarding no
            X11Forwarding no

3. restart ssh service: sudo systemctl restart ssh

---

Share hard drive:
https://askubuntu.com/questions/742487/sharing-an-external-hard-drive-between-users
For ntfs: 
> sudo nano /etc/fstab
Put at the end:
UUID=0000000000000000 /media/new_drive ntfs-3g defaults,uid=1000,umask=0022 0 0

Check "linux.md" to get more details about groups.
