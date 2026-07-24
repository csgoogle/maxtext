# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installs extra dependencies from a requirements file using uv.

This script is designed to install dependencies specified in 'dependencies/extra_deps/post_train_*.txt'.
It first ensures 'uv' is installed and then uses it to install the packages listed in the requirements file.
"""

import os
import shutil
import subprocess
import sys


def ensure_cpp20_compiler():
  """Ensures GCC/G++ >= 11 (e.g. gcc-12/gcc-11) is available and configured for building vLLM."""
  if sys.platform != "linux":
    return
  try:
    res = subprocess.run(["gcc", "-dumpversion"], capture_output=True, text=True, check=False)
    major_ver = int(res.stdout.strip().split(".")[0])
    if major_ver >= 11:
      return
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  if shutil.which("gcc-12") and shutil.which("g++-12"):
    os.environ["CC"] = "gcc-12"
    os.environ["CXX"] = "g++-12"
    print("Using pre-installed C++20 compiler: CC=gcc-12 CXX=g++-12")
    return
  if shutil.which("gcc-11") and shutil.which("g++-11"):
    os.environ["CC"] = "gcc-11"
    os.environ["CXX"] = "g++-11"
    print("Using pre-installed C++20 compiler: CC=gcc-11 CXX=g++-11")
    return

  is_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
  has_sudo = shutil.which("sudo") is not None
  if (is_root or has_sudo) and shutil.which("apt-get"):
    try:
      print("Ensuring GCC 12/11 for vLLM C++20 compilation...")
      prefix = [] if is_root else ["sudo", "-E"]
      if os.path.exists("/etc/os-release"):
        with open("/etc/os-release", "r", encoding="utf-8") as f:
          os_rel = f.read()
        if "bullseye" in os_rel and not os.path.exists("/etc/apt/sources.list.d/bookworm.list"):
          sources_str = "deb http://deb.debian.org/debian bookworm main\n"
          bookworm_cmd = [
              "sh",
              "-c",
              f'echo "{sources_str}" > /etc/apt/sources.list.d/bookworm.list',
          ]
          subprocess.run(prefix + bookworm_cmd, check=False)

      subprocess.run(prefix + ["apt-get", "update", "-y"], check=False)
      install_bookworm = prefix + [
          "apt-get",
          "install",
          "-y",
          "--no-install-recommends",
          "-t",
          "bookworm",
          "gcc-12",
          "g++-12",
          "build-essential",
          "cmake",
          "ninja-build",
      ]
      subprocess.run(install_bookworm, check=False)

      install_regular = prefix + [
          "apt-get",
          "install",
          "-y",
          "--no-install-recommends",
          "gcc-12",
          "g++-12",
          "gcc-11",
          "g++-11",
          "build-essential",
          "cmake",
          "ninja-build",
      ]
      subprocess.run(install_regular, check=False)

      if shutil.which("gcc-12") and shutil.which("g++-12"):
        os.environ["CC"] = "gcc-12"
        os.environ["CXX"] = "g++-12"
        print("Successfully configured C++20 compiler: CC=gcc-12 CXX=g++-12")
      elif shutil.which("gcc-11") and shutil.which("g++-11"):
        os.environ["CC"] = "gcc-11"
        os.environ["CXX"] = "g++-11"
        print("Successfully configured C++20 compiler: CC=gcc-11 CXX=g++-11")
      else:
        print("Warning: gcc-12/gcc-11 binary not found after apt-get execution.")
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Warning: Failed to install GCC 12/11 via apt-get: {e}")


def main():
  """
  Installs extra dependencies specified in 'dependencies/extra_deps/post_train_*.txt' using uv.
  It executes 'uv pip install -r <path_to_extra_deps.txt> --resolution=lowest'.
  """
  os.environ["VLLM_TARGET_DEVICE"] = "tpu"
  os.environ["UV_TORCH_BACKEND"] = "cpu"
  ensure_cpp20_compiler()

  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  github_deps_path = os.path.join(repo_root, "dependencies", "extra_deps", "post_train_github_deps.txt")
  # Ensure 'uv' is installed in the environment
  try:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True, capture_output=True)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Warning: error installing uv via pip: {e}")

  github_deps_command = [
      "uv",
      "pip",
      "install",
      "--python",
      sys.executable,
      "-r",
      str(github_deps_path),
      "--no-deps",
      "--no-build-isolation",
  ]

  local_vllm_install_command = [
      "uv",
      "pip",
      "install",
      "--python",
      sys.executable,
      f"{repo_root}/maxtext/integration/vllm",  # MaxText on vllm installations
      "--no-deps",
  ]

  try:
    # Run the command to install Github dependencies
    print(f"Installing Github dependencies: {' '.join(github_deps_command)}")
    _ = subprocess.run(github_deps_command, check=True, capture_output=True, text=True, env=os.environ)
    print("Github dependencies installed successfully!")

    # Run the command to install the MaxText vLLM directory
    print(f"Installing MaxText vLLM dependency: {' '.join(local_vllm_install_command)}")
    _ = subprocess.run(local_vllm_install_command, check=True, capture_output=True, text=True, env=os.environ)
    print("MaxText vLLM dependency installed successfully!")
  except subprocess.CalledProcessError as e:
    print("Failed to install extra dependencies.")
    print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
    print("--- Stderr ---")
    print(e.stderr)
    print("--- Stdout ---")
    print(e.stdout)
    sys.exit(e.returncode)
  except (OSError, FileNotFoundError) as e:
    print(f"An OS-level error occurred while trying to run uv: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
