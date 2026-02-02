from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("""
    typedef int... mode_t;
    typedef struct py_dfs py_dfs_t;

    int daos_init(void);
    int daos_fini(void);

    py_dfs_t *new_py_dfs_t(void);
    void free_py_dfs_t(py_dfs_t *hdl);
    
    int py_dfs_open(char *pool, char *cont, py_dfs_t *hdl);
    int py_dfs_close(py_dfs_t *hdl);

    int py_dfs_mkdir(py_dfs_t *hdl, char *parent, char *name);
    int py_dfs_get_mode(py_dfs_t *hdl, char *path, mode_t *mode);
""")

ffibuilder.set_source(
    "py_dfs",                      
    """
    #include <errno.h>
    #include <fcntl.h>
    #include <sys/stat.h>

    #include <daos.h>
    #include <daos_fs.h>
    #include <gurt/common.h>

    struct py_dfs {
        daos_handle_t  poh;
        daos_handle_t  coh;
        dfs_t         *dfs;
    };

    typedef struct py_dfs py_dfs_t;

    
    py_dfs_t *new_py_dfs_t(void) {
        py_dfs_t *p = NULL;
        D_ALLOC_PTR(p);
        return p;
    }

    void free_py_dfs_t(py_dfs_t *hdl) { D_FREE(hdl); }

    int py_dfs_open(char *pool, char *cont, py_dfs_t *hdl) {
        int rc = daos_pool_connect(pool, NULL, DAOS_PC_RW, &hdl->poh, NULL, NULL);
        if (rc) {
           return rc;
        }
        rc = daos_cont_open(hdl->poh, cont, DAOS_COO_RW, &hdl->coh, NULL, NULL);
        if (rc != 0) goto err_cont;

        rc = dfs_mount(hdl->poh, hdl->coh, O_RDWR, &hdl->dfs);
        if (rc != 0) goto err_mount;

        return 0;

err_mount:
        daos_cont_close(hdl->coh, NULL);
err_cont:
        daos_pool_disconnect(hdl->poh, NULL);
        return rc;
    }

    int py_dfs_close(py_dfs_t *hdl) {
        int rc = dfs_umount(hdl->dfs);
        if (rc) return rc;

        rc = daos_cont_close(hdl->coh, NULL);
        if (rc) return rc;

        return daos_pool_disconnect(hdl->poh, NULL);
    }


    int py_dfs_mkdir(py_dfs_t *hdl, char *parent, char *name) {
        dfs_obj_t *obj;
        int rc = dfs_lookup(hdl->dfs, parent, O_RDWR, &obj, NULL, NULL);
        if (rc) {
            D_ERROR("dfs_lookup error: name='%s', rc=%d", parent, rc);
            return rc;
        }

        mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH | S_IFDIR;
        rc = dfs_mkdir(hdl->dfs, obj, name, mode, 0);
        dfs_release(obj);
        return rc;
    }

    int py_dfs_get_mode(py_dfs_t *hdl, char *path, mode_t *mode) {
        dfs_obj_t *obj;
        int rc = dfs_lookup(hdl->dfs, path, O_RDONLY, &obj, mode, NULL);
        if (rc) {
            D_ERROR("dfs_lookup error: path='%s', rc=%d", path, rc);
            return rc;
        }
        return dfs_release(obj);
    }

    """,
    libraries=["daos", "dfs"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
