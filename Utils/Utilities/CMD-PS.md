## Good packages
- Chocolatey: Allow to refresh env variables with command 'refreshenv'.

## Configuration

- Print the environment variables

```bash
set # CMD
ls env: # Powershell
```

- Set permanently a environment variable
```bash
setx <VARIABLE> <VALUE>
```

- Print a environment variable
```bash
echo %<VARIABLE>% # CMD
$env:<VARIABLE> # Powershell
```

- Append <NEWVALUE> to a environment variable
```bash
setx <VARIABLE> "$env:<VARIABLE><NEWVALUE>" # Powershell
```