
In shell often is shown:

user_name@static_hostname

- Command to get user name: `who` or `whoami`
    - To list all users: `cat /etc/passwd`
    - Normal users (those that are logged at the beggining of the session) have and UID that usually starts with 1000.
        - Check [this link](https://devconnected.com/how-to-list-users-and-groups-on-linux/#:~:text=In%20order%20to%20list%20users,navigate%20within%20the%20username%20list.) for more details.
- Command to get hostname : `hostname`
    - When making SSH, this is the IP adress.


---

Groups and User:

- Check [this link](https://devconnected.com/how-to-list-users-and-groups-on-linux/#:~:text=In%20order%20to%20list%20users,navigate%20within%20the%20username%20list.) for more details.

- list users:
cat /etc/passwd

- list groups:

cat /etc/group


---

Increase swap:
https://www.cloudsigma.com/adding-swap-space-on-ubuntu-20-04-a-tutorial/

---

open ports:

sudo lsof -i -P -n | grep LISTEN
 