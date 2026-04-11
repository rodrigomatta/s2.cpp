#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(S2_STATIC)
#    define S2_Export
#  elif defined(S2_LIBRARY)
#    define S2_Export __declspec(dllexport)
#  else
#    define S2_Export __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define S2_Export __attribute__((visibility("default")))
#else
#  define S2_Export
#endif
