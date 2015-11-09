#include "wincompat.h"
#include <cmath>

double round(double r){
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}
