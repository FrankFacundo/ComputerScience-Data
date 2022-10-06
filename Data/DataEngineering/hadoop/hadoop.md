# Hadoop commands

## Copy files or directories 

Copy files or directories between different edge nodes

```bash
hadoop distcp \
-D ipc.client.fallback-to-simple-auth-allowed=true \
-overwrite \
-delete \
-pb \
$PATH_SOURCE
$PATH_DEST

# PATH_SOURCE = webhdfs://node:port/remote_path_source
# PATH_DEST   = hdfs://node/remote_path_dest
```

