/* Diagnostic-only LD_PRELOAD hook. Preserve abort; capture native callers first.
 * Best effort: hidden/internal abort calls or direct signals may bypass this hook.
 * Backtrace is collected before abort, not from a signal handler. No GPU API calls.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static __thread int tracing;

static void dump_trace(const char *reason) {
    if (tracing) return;
    tracing = 1;
    const char *directory = getenv("ROBOT_PRACTICE_NATIVE_TRACE_DIR");
    char filename[4096];
    int fd = -1;
    if (directory) {
        int n = snprintf(filename, sizeof(filename), "%s/native_%ld.log",
                         directory, (long)getpid());
        if (n > 0 && n < (int)sizeof(filename))
            fd = open(filename, O_WRONLY | O_CREAT | O_APPEND, 0600);
    }
    if (fd < 0) fd = STDERR_FILENO;
    dprintf(STDERR_FILENO, "[NATIVE-ABORT] pid=%ld reason=%s\n", (long)getpid(), reason);
    dprintf(fd, "[NATIVE-ABORT] pid=%ld reason=%s\n", (long)getpid(), reason);
    void *frames[96];
    int count = backtrace(frames, 96);
    for (int i = 0; i < count; ++i) {
        Dl_info info = {0};
        if (dladdr(frames[i], &info)) {
            dprintf(fd, "#%d pc=%p object=%s offset=0x%lx symbol=%s\n", i, frames[i],
                    info.dli_fname ? info.dli_fname : "?",
                    (unsigned long)((uintptr_t)frames[i] - (uintptr_t)info.dli_fbase),
                    info.dli_sname ? info.dli_sname : "?");
        }
    }
    backtrace_symbols_fd(frames, count, fd);
    if (fd != STDERR_FILENO) {
        fsync(fd);
        close(fd);
    }
}

__attribute__((noreturn)) void abort(void) {
    void (*original)(void) = dlsym(RTLD_NEXT, "abort");
    dump_trace("abort");
    if (original) original();
    _exit(134);
}

__attribute__((noreturn)) void __assert_fail(const char *assertion, const char *file,
                                            unsigned int line, const char *function) {
    void (*original)(const char *, const char *, unsigned int, const char *) =
        dlsym(RTLD_NEXT, "__assert_fail");
    char reason[2048];
    snprintf(reason, sizeof(reason), "assertion=%s file=%s line=%u function=%s",
             assertion, file, line, function ? function : "?");
    dump_trace(reason);
    if (original) original(assertion, file, line, function);
    abort();
}
