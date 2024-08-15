# SSH

The public part of the key pair is just like a house lock.
The private part of the key pair is just like a house key.

### Add two keys (ex. Github and Gitlab)

#### Linux

- Create both SSH keys (in ~/.ssh)
- in the same directory create the file `config`

Then add and adapt the following lines to the file:

```
Host 192.168.1.143
  HostName 192.168.1.143
  User frank

Host gitlab.com
  HostName gitlab.com
  User git
  IdentityFile ~/.ssh/id_rsa_gitlab
```
