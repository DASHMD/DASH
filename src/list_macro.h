#pragma once
#ifndef LIST_MACRO_H
#define LIST_MACRO_H

#define LISTMAP(from, to, x, xs, fn) listmap< from, to >(xs, [&] (const from x) {return fn;})
#define LISTMAPTEST(from, to, x, xs, proc, test) listmaptest< from, to >(xs, [&] (from x) {return proc;}, [&] (from x){ return test;})

#define LISTMAPREF(from, to, x, xs, fn) listmapref< from, to >(xs, [&] (const from &x) {return fn;})
#define LISTMAPREFTEST(from, to, x, xs, proc, test) listmapreftest< from, to >(xs, [&] (from &x) {return proc;}, [&] (from &x){ return test;})

//#define LISTMAPPY(from, x, xs, fn) listmapPy< from >(xs, [&] (const from x) {return fn;})
//#define LISTMAPPYREF(from, x, xs, fn) listmapPyRef< from >(xs, [&] (const from &x) {return fn;})
//#define LISTMAPFROMPY(to, x, xs, fn) listmapFromPy< to >(xs, [&] (PyObject *x) {return fn;})
//stuff for c arrays
//#define ARRAYMAPPY(from, x, xs, len, fn) arraymapPy< from > (xs, len, [&] (const from x) {return fn;})

#include <vector>
#include <functional>
#include <iostream>

template<class A, class B>
std::vector<B> listmap(const std::vector<A> &src, std::function<B (const A)> fn) {
    std::vector<B> bs;
    bs.reserve(src.size());
    for (auto it = src.begin(); it != src.end(); it++) {
        bs.push_back(fn(*it));
    }
    return bs;
}

template<class A, class B>
std::vector<B> listmaptest(std::vector<A> &src, std::function<B (A)> proc,
                           std::function<bool (A)> test) {
    std::vector<B> bs;
    for (auto it = src.begin(); it != src.end(); it++) {
        if (test(*it)) {
            bs.push_back(proc(*it));
        }
    }
    return bs;
}

template<class A, class B>
std::vector<B> listmapref(const std::vector<A> &src, std::function<B (const A&)> fn) {
    std::vector<B> bs;
    bs.reserve(src.size());
    for (auto it = src.begin(); it != src.end(); it++) {
        bs.push_back(fn(*it));
    }
    return bs;
}

template<class A, class B>
std::vector<B> listmapreftest(std::vector<A> &src, std::function<B (A&)> proc,
                              std::function<bool (A&)> test) {
    std::vector<B> bs;
    for (auto it = src.begin(); it != src.end(); it++) {
        if (test(*it)) {
            bs.push_back(proc(*it));
        }
    }
    return bs;
}

/*
template<class A>
PyObject *listmapPy(const std::vector<A> &src, std::function<PyObject *(const A)> fn) {
    PyObject *l = PyList_New(src.size());
    for (unsigned int i=0; i<src.size(); i++) {
        PyList_SetItem(l, i, fn(src[i]));
    }
    return l;
}

template<class A>
PyObject *listmapPyRef(const std::vector<A> &src, std::function<PyObject *(const A &)> fn) {
    PyObject *l = PyList_New(src.size());
    for (unsigned int i=0; i<src.size(); i++) {
        PyList_SetItem(l, i, fn(src[i]));
    }
    return l;
}

template<class A>
std::vector<A> listmapFromPy(PyObject *src, std::function<A (PyObject *)> fn) {
    int size = PyList_Size(src);
    if (size == -1) {
        cout << "Trying to map from bad python list" << endl;
    }
    std::vector<A> as;
    as.reserve(size);
    for (int i=0; i<size; i++) {
        PyObject *o = PyList_GetItem(src, i);
        as.push_back(fn(o));
    }
    return as;
}

template<class A> //A must be of pointer type
PyObject *arraymapPy(const A *src, unsigned int len, std::function<PyObject *(const A)> fn) {
    PyObject *l = PyList_New(len);

    for (unsigned int i=0; i<len; i++) {
        PyList_SetItem(l, i, fn(src[i]));
    }
    return l;
}
*/

#endif
