# Python

## Configuration

- Change version

```bash
alias python='/usr/bin/python3.4'
. ~/.bashrc
python --version
```

- Check library version

```bash
pip show packageName
```

- To check configuration of pip

```bash
pip config -v list
```

- Config file

`pip.ini`

```text
[global]
timeout = 60
index = [link source pypi]
index-url = [link source pypi]
cert = [path of certificate file]
```

## Tools

Formatters: Yapf (modulable), Black (static)
Linter: pylint
