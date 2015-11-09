#if !defined _HEADER_WIN_COMPAT_20140627_INCLUDED_
#define _HEADER_WIN_COMPAT_20140627_INCLUDED_

typedef unsigned int uint;
#define snprintf _snprintf

double round(double r);

#define __builtin_popcount __popcnt 
#define __builtin_popcountl __popcnt

#endif //_HEADER_WIN_COMPAT_20140627_INCLUDED_


