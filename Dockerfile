FROM registry.fedoraproject.org/fedora:44

COPY entrypoint/*.sh /usr/local/bin/
COPY entrypoint/entrypoint.sh /
COPY python/requirements.txt /requirements.txt

RUN  --mount=type=cache,target=/var/cache/libdnf5 \
     --mount=type=cache,target=/var/lib/dnf \
     echo "keepcache=True" >> /etc/dnf/dnf.conf && \
     dnf upgrade -y && \
     dnf install -y \
                   gcc \
                   gcc-g++ \
                   cmake \
                   ninja-build \
                   git \
                   glib \
                   glib2-devel \
                   glfw \
                   meson \
                   ninja \
                   python3-glfw \
                   python3-numpy \
                   python3-pillow \
                   python3-pyopengl \
                   python3-pytest \
                   python3-virtualenv \
                   ruff \
                   uv \
                   tmux \
                   which \
                   libglvnd-gles \
                   libXcomposite \
                   libXcursor \
                   libXdamage \
                   libXfixes \
                   libXft \
                   libXi \
                   libXinerama \
                   libXmu \
                   libXrandr \
                   libXrender \
                   libXres \
                   libXtst \
                   libXv \
                   libXxf86vm \
                   mesa-dri-drivers  \
                   mesa-libGLU-devel \
                   libwayland-client \
                   libwayland-cursor \
                   libwayland-egl \
                   libxkbcommon \
                   libdecor \
                   mesa-libGL-devel \
                   mesa-libEGL-devel \
                   libX11-devel \
                   libXrandr-devel \
                   libXi-devel \
                   libXcursor-devel \
                   libXinerama-devel \
                   wayland-devel \
                   wayland-protocols-devel \
                   libxkbcommon-devel \
                   extra-cmake-modules ; \
    python3 -m venv --system-site-packages /venv/ && \
    export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
    source /venv/bin/activate && \
    dnf install -y libatomic && \
    uv pip install setuptools ty && \
    uv pip install --python "$(which python)" -r /requirements.txt && \
    rm /requirements.txt

# Bake the C examples and the Python port from the source so a built image is
# validated at build time and a pulled image is runnable standalone. `make
# shell` bind-mounts the live repo over /opengl, which SHADOWS this baked copy;
# the entrypoint then re-builds the C (incremental) and re-installs the Python
# package against the live mount, so the two always match the mounted source.
# The CMake build fetches GLFW 3.4 / GLEW 2.2.0 / GLUS at configure time and
# writes the ExampleNN binaries into Binaries/ and the static libs into lib/.
# .dockerignore keeps .git, the host build trees, Binaries/ and lib/ out of the
# COPY so the build populates fresh ones.
#
# The build TREE lives at /opt/ogl-build, OUTSIDE the /opengl mount, on purpose:
# `make shell` bind-mounts the host repo over /opengl, and the host's own build/
# (a CMakeCache configured for the host path) would otherwise collide with a
# build tree under /opengl and cmake would refuse ("source does not match").
# Because the mount is at the same /opengl path this cache was configured
# against, the entrypoint's `cmake --build /opt/ogl-build` is a fast incremental
# relink against the live source, re-emitting any binary missing from the
# (now host-mounted) Binaries/.
COPY . /opengl
RUN cmake -S /opengl -B /opt/ogl-build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/ogl-build && \
    export VIRTUAL_ENV_DISABLE_PROMPT=1 && \
    source /venv/bin/activate && \
    cd /opengl/python && \
    uv pip install --no-deps --no-index --no-build-isolation -e . \
        --python "$(which python)"

ENTRYPOINT ["/entrypoint.sh"]
