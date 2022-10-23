Commands of `Dockerfile`
Ref: https://docs.docker.com/engine/reference/builder/

- FROM: ex. `FROM ubuntu:20.04`
- ENV: environment variables 
- RUN: Execute console commands 
- WORKDIR: the default dir
- ADD: copy dirs ```ADD hom* /mydir/``` ->hom is source and mydir is destination.
- CMD: There can only be one `CMD` instruction in a `Dockerfile`. If you list more than one `CMD` then only the last `CMD` will take effect.
**The main purpose of a `CMD` is to provide defaults for an executing container.**
- EXPOSE: The `EXPOSE` instruction informs Docker that the container listens on the specified network ports at runtime. You can specify whether the port listens on TCP or UDP, and the default is TCP if the protocol is not specified.