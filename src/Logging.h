#pragma once
#ifndef LOGGING_H
#define LOGGING_H

#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct ReturnException : std::exception
{
    virtual char const* what() const throw() { return "Stopping program."; }
};

struct AssertFailedException : std::exception
{
    virtual char const* what() const throw() { return "Assert failed."; }
};

// Strip path from filename
#define __FILENAME__ std::strrchr(__FILE__, '/') ? \
                     std::strrchr(__FILE__, '/') + 1 : \
                     __FILE__

// Write debug output only if DEBUG is defined
#ifdef DEBUG
    #define DBG 1
#else
    #define DBG 0
#endif

#define cudaDebug(fmt, ...) \
    do { if (DBG) printf("DEBUG: " fmt " (in %s:%d:%s())\n", \
                         ##__VA_ARGS__, \
                         __FILE__, __LINE__, __func__); \
    } while (false)

#define cudaMessage(fmt, ...) \
    do { printf(fmt, ##__VA_ARGS__); } while(false)

#define mdDebug(fmt, ...) \
    do { if (DBG) fprintf(stdout, "DEBUG: " fmt " (in %s:%d:%s())\n", \
                          ##__VA_ARGS__, \
                          __FILENAME__, __LINE__, __func__); \
    } while(false)

#define mdMessage(fmt, ...) \
    do { fprintf(stdout, fmt, ##__VA_ARGS__); } while (false)

#define mdWarning(fmt, ...) \
    do { fprintf(stderr, "WARNING: " fmt " (in %s:%d)\n", \
                 ##__VA_ARGS__, \
                 __FILENAME__, __LINE__); \
    } while(false)

#define mdError(fmt, ...) \
    do { fprintf(stderr, "ERROR: " fmt " (in %s:%d)\n", \
                 ##__VA_ARGS__, \
                 __FILENAME__, __LINE__); \
         throw ReturnException(); \
    } while(false)

#define mdCritical(exception, fmt, ...) \
    do { fprintf(stderr, "ERROR: " fmt " (in %s:%d)\n", \
                 ##__VA_ARGS__, \
                 __FILENAME__, __LINE__); \
         throw exception; \
    } while(false)

#define mdFatal(exitCode, fmt, ...) \
    do { fprintf(stderr, "FATAL: " fmt " (in %s:%d)\n", \
                 ##__VA_ARGS__, \
                 __FILENAME__, __LINE__); \
         std::exit(1); \
    } while(false)

#define mdAssume(test, fmt, ...) \
    do { if (!(test)) { fprintf(stderr, "WARNING: In %s(): Assumption " \
                                        #test " failed: " \
                                        fmt " (%s:%d)\n", \
                                __func__, ##__VA_ARGS__, \
                                __FILENAME__, __LINE__); } \
    } while (false)

#define mdAssert(test, fmt, ...) \
    do { if (!(test)) { fprintf(stderr, "ERROR: In %s(): Assertion " \
                                        #test " failed: " \
                                        fmt " (%s:%d)\n", \
                                __func__, ##__VA_ARGS__, \
                                __FILENAME__, __LINE__); \
                        throw AssertFailedException(); } \
    } while (false)

#endif
