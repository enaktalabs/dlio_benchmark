import os
import errno

from pathlib import Path
from py_dfs import ffi, lib

def c_str(s: str) -> bytes:
    if not s:
        return ffi.NULL

    return s.encode('utf-8')

def check_rc(rc, msg):
    if rc != 0:
        raise RuntimeError(f"{msg} failed: {rc}")


def mkdirall(hdl, path):
    dirs = list(Path(path).parts)
    if not dirs:
        raise RuntimeError(f"invalid path: {path}")

    base = dirs.pop(0)
    for name in dirs:
        print(f"creating dir {name} in {base}")
        rc = lib.py_dfs_mkdir(hdl, c_str(base),  c_str(name))
        if rc != 0 and rc != errno.EEXIST:
            raise RuntimeError(f"could not create directory '{name}' in '{base}'")
        base = os.path.join(base, name)
    
    
check_rc(lib.daos_init(), "daos_init")

hdl = lib.new_py_dfs_t()

pool = c_str("pp1")
cont = c_str("cc")

check_rc(lib.py_dfs_open(pool, cont, hdl), "py_dfs_open")

mkdirall(hdl, "/a/b/c/tomato")
# :check_rc(lib.py_dfs_mkdir(hdl, c_str(dir), c_str(name)), "mkdir")

check_rc(lib.py_dfs_close(hdl), "py_dfs_close")
lib.free_py_dfs_t(hdl)

