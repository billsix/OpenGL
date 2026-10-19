.DEFAULT_GOAL := shell

# Autodetect the container runtime (podman preferred, docker fallback).
CONTAINER_CMD ?= $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)
CONTAINER_NAME = opengl

# This Makefile lives at the repo root. The container mounts the whole repo at
# /opengl and builds BOTH halves there: the C examples (cmake -> ExampleNN
# binaries in Binaries/) and the Python port (editable install of python/). The
# C demos load assets by bare filename from Binaries/ as cwd; the Python port
# reads the same Binaries/ via MCNOPPER_MEDIA.
REPO_ROOT := $(CURDIR)
REPO_MOUNT := /opengl
PKG_DIR := $(REPO_MOUNT)/python
MEDIA_ENV := -e MCNOPPER_MEDIA=$(REPO_MOUNT)/Binaries

# Extra flags for every container `run`. Auto-set when running nested inside a
# runClaudeInContainer/runCrushInContainer sandbox (which exports NESTED_PODMAN=1),
# so podman-in-podman works; empty on a normal host. On `run` lines only.
PODMAN_RUN_FLAGS ?= $(if $(filter 1,$(NESTED_PODMAN)),--cgroups=disabled)

TMUX_FILE := $(HOME)/.tmux.conf
TMUX_REAL_PATH := $(shell readlink -f $(TMUX_FILE))
TMUX_MOUNT := $(shell if [ -f $(TMUX_REAL_PATH) ]; then echo "-v $(TMUX_REAL_PATH):/root/.tmux.conf:Z" ; fi)

GITCONFIG_FILE := $(HOME)/.gitconfig
GITCONFIG_REAL_PATH := $(shell readlink -f $(GITCONFIG_FILE))
GITCONFIG_MOUNT := $(shell if [ -f $(GITCONFIG_REAL_PATH) ]; then echo "-v $(GITCONFIG_REAL_PATH):/root/.gitconfig:Z" ; fi)

# Mount the repo root (so the C sources, shaders and Binaries/ are reachable)
# plus the live entrypoint scripts over their in-image paths (so edits take
# effect without a rebuild). `--security-opt label=disable` instead of a :z/:Z
# relabel: the mounted tree includes the ~540 MB Binaries/, expensive to relabel
# and shared with the host, so SELinux confinement is turned off for the (local,
# dev) container rather than relabel the repo.
FILES_TO_MOUNT = --security-opt label=disable \
                 -v $(REPO_ROOT):$(REPO_MOUNT) \
                 -v $(CURDIR)/entrypoint/entrypoint.sh:/entrypoint.sh \
                 -v $(CURDIR)/entrypoint/shell.sh:/usr/local/bin/shell.sh \
                 -v $(CURDIR)/entrypoint/loadpackages.sh:/usr/local/bin/loadpackages.sh \
                 $(TMUX_MOUNT) \
                 $(GITCONFIG_MOUNT)

USE_X = -e DISPLAY=$(DISPLAY) \
        -v /tmp/.X11-unix:/tmp/.X11-unix

WAYLAND_FLAGS_FOR_CONTAINER = -e "XDG_SESSION_TYPE=wayland" \
                              -e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY}" \
                              -e "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}" \
                              -e "PYGLFW_LIBRARY=/usr/lib64/libglfw.so.3" \
                              -v "${XDG_RUNTIME_DIR}:${XDG_RUNTIME_DIR}"

# Configure + build the C examples against the live mount. The build tree is
# /opt/ogl-build, OUTSIDE the mount, so the host's own build/ cache cannot
# collide with it; the image baked + configured it, so this is a fast
# incremental relink. CMake writes the ExampleNN binaries into Binaries/.
C_BUILD_PRELUDE = cmake -S $(REPO_MOUNT) -B /opt/ogl-build -G Ninja \
	-DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build /opt/ogl-build

# Editable-install the Python port against the live mount. The `\$$(which python)`
# is escaped so it evaluates inside the container (after the venv activate), not
# on the host.
PY_INSTALL_PRELUDE = source /venv/bin/activate && cd $(PKG_DIR) && \
	uv pip install --no-deps --no-index --no-build-isolation -e . --python \$$(which python) >/dev/null


.PHONY: all
all: image ## Build the container image

.PHONY: image
image: ## Build the podman image (builds the C examples + installs the Python port)
	$(CONTAINER_CMD) build \
                         -t $(CONTAINER_NAME) \
                         .

# `shell` forwards BOTH X11 and Wayland, so the container has both display
# backends. A Python demo (`cd python && mcnopper 5`) uses Wayland natively; a
# C demo must use X11 because GLUS+GLEW resolves GL through GLX and cannot on an
# EGL/Wayland context — so the shell provides a `run-c <N>` helper that launches
# the C example with WAYLAND_DISPLAY unset (→ X11/XWayland). GLFW auto-picks the
# backend from WAYLAND_DISPLAY, which is process-global, so a *bare*
# `./ExampleNN` under Wayland can't draw; use `run-c <N>` (the shell prints the
# hint). `make run-c N=…` does the same from outside the shell.
.PHONY: shell
shell: ## Get a shell in an ephemeral container (repo root; both X11 + Wayland; run-c helper for C)
	$(CONTAINER_CMD) run $(PODMAN_RUN_FLAGS) -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(MEDIA_ENV) \
		$(USE_X) \
		$(WAYLAND_FLAGS_FOR_CONTAINER) \
		$(CONTAINER_NAME) \
		/usr/local/bin/shell.sh

# run/list/run-c/build-c override the image ENTRYPOINT (the health check) with
# /bin/bash so the command after it runs -- same as `shell`.
.PHONY: run
run: ## Run a Python demo in the container: make run N=13
	@if [ -z "$(N)" ]; then echo "usage: make run N=<demo-number 1..52>"; exit 1; fi
	$(CONTAINER_CMD) run $(PODMAN_RUN_FLAGS) -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(MEDIA_ENV) \
		$(USE_X) \
		$(WAYLAND_FLAGS_FOR_CONTAINER) \
		$(CONTAINER_NAME) \
		-c "$(PY_INSTALL_PRELUDE) && mcnopper $(N)"

.PHONY: list
list: ## List all 52 Python demos with their one-line descriptions
	$(CONTAINER_CMD) run $(PODMAN_RUN_FLAGS) -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(MEDIA_ENV) \
		$(CONTAINER_NAME) \
		-c "$(PY_INSTALL_PRELUDE) && mcnopper list"

.PHONY: check
check: ## Import every Python demo module against Binaries/ (no display)
	$(CONTAINER_CMD) run $(PODMAN_RUN_FLAGS) --rm \
		$(FILES_TO_MOUNT) \
		$(MEDIA_ENV) \
		$(CONTAINER_NAME)

# Read-only type gate: ty (correctness) over the mcnopper package -- which holds
# both the shared modules AND the demos (exampleNN submodules) -- plus
# tools/check_local_annotations.py (completeness -- the "annotate every binding"
# rule ty can't enforce). Editable-install first so `mcnopper` resolves. Every
# step runs and the target fails if ANY failed. `$$s`/`$$(...)` are literal `$`.
.PHONY: type-check
type-check: ## Type-check the Python port: ty + annotate-every-binding checker
	$(CONTAINER_CMD) run $(PODMAN_RUN_FLAGS) --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(CONTAINER_NAME) \
		-c 'source /venv/bin/activate && cd $(PKG_DIR) && \
		    uv pip install --no-deps --no-index --no-build-isolation -e . \
		        --python "$$(which python)" >/dev/null; \
		    s=0; \
		    ty check src/mcnopper || s=1; \
		    cd $(REPO_MOUNT) && \
		    python tools/check_local_annotations.py --include-module \
		        python/src/mcnopper || s=1; \
		    exit $$s'

.PHONY: format
format: ## Format the Python port (ruff check --fix + ruff format) in the container
	$(CONTAINER_CMD) run $(PODMAN_RUN_FLAGS) --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(CONTAINER_NAME) \
		-c 'cd $(REPO_MOUNT) && bash entrypoint/format.sh'

.PHONY: build-c
build-c: ## Build the C examples in the container (into Binaries/)
	$(CONTAINER_CMD) run $(PODMAN_RUN_FLAGS) --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(CONTAINER_NAME) \
		-c "$(C_BUILD_PRELUDE)"

# run-c forwards both backends (like `shell`) but launches the C binary with
# WAYLAND_DISPLAY unset, so GLFW uses X11/GLX — GLUS+GLEW cannot resolve GL on an
# EGL/Wayland context (glewInit returns GLEW_ERROR_NO_GLX_DISPLAY), so a C demo
# under Wayland opens a window and then cannot draw. On a Wayland host X11 is
# XWayland, where GLX works. (The Python port uses PyOpenGL, which handles EGL,
# so `run` stays on Wayland.)
.PHONY: run-c
run-c: ## Run a C example on your display: make run-c N=13
	@if [ -z "$(N)" ]; then echo "usage: make run-c N=<example-number 1..52>"; exit 1; fi
	$(CONTAINER_CMD) run $(PODMAN_RUN_FLAGS) -it --rm \
		--entrypoint /bin/bash \
		$(FILES_TO_MOUNT) \
		$(USE_X) \
		$(WAYLAND_FLAGS_FOR_CONTAINER) \
		$(CONTAINER_NAME) \
		-c "$(C_BUILD_PRELUDE) && cd $(REPO_MOUNT)/Binaries && env -u WAYLAND_DISPLAY ./Example$$(printf '%02d' $(N))"

.PHONY: clean
clean: ## Remove C and Python build artefacts
	rm -rf build python/build python/dist python/*.egg-info python/src/*.egg-info

.PHONY: image-export
image-export: ## Export the OCI image to a tarball
	$(CONTAINER_CMD) save $(CONTAINER_NAME) -o $(CONTAINER_NAME)-$(shell date +%m-%d-%Y_%H-%M-%S).tar

.PHONY: image-import
image-import: ## Import a tarball as the OCI image: make image-import FILE=foo.tar
	$(CONTAINER_CMD) load -i $(FILE)

.PHONY: help
help:
	@grep --extended-regexp '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
