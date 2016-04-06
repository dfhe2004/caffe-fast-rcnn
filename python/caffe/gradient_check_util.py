
import numpy
import logging

from IPython import embed


def reldiff(a, b):
    diff = numpy.sum(numpy.abs(a - b))
    norm = numpy.sum(numpy.abs(a))
    if diff == 0:
        return 0
    reldiff = diff  / norm
    return reldiff


def _pp(name,arr):
    return '------ %s ------\n%s\n'%(name, arr.flatten())

def check_gradient(layer, bottom, top, skip_bottom=None, skip_blobs=None, numeric_eps=1e-3, check_eps=1e-3, seed=1701 ):
    assert len(top)==1, 'only support one top'

    skip_bottom  = set() if skip_bottom is None else set(skip_bottom)
    skip_blobs    = set() if skip_blobs is None else set(skip_blobs)

    blobs  =  [ e for i,e in enumerate(bottom) if not i in skip_bottom ]
    blobs.extend([ e for i,e in enumerate(layer.blobs) if not i in skip_blobs])
    blobs_data  = [ e.data.copy() for e in blobs ]

    rng = numpy.random.RandomState(seed)
    
    p = rng.randn(*list(top[0].data.shape)) + 0.1 
    layer.SetUp(bottom,top)
    layer.Forward(bottom,top)

    loss = (top[0].data*p).sum()
    top[0].diff[...] = p

    layer.Backward(top, [True,]*len(bottom), bottom )
    sym_grads = [e.diff.copy() for e in blobs]

    #embed()
    #-- numeric diff
    for iG, grad in enumerate(sym_grads):
        
        #-- recover blobs
        for iB, blob in enumerate(blobs):
            blob.data[...] = blobs_data[iB]
            blob.diff[...] = 0

        _grad = []
        _diff = numpy.zeros(grad.size)
        
        for iD in xrange(grad.size):
            _diff = _diff.reshape(grad.size)
            if iD>0:
                _diff[iD-1] = 0
            _diff[iD]   = numeric_eps
            _diff = _diff.reshape(grad.shape)

            #-- 
            blobs[iG].data[...] = blobs_data[iG] + _diff            
            
            layer.Forward(bottom,top)
            lossh =  (top[0].data*p).sum()
            val = (lossh-loss)/numeric_eps
            _grad.append(val)                        
        
        _grad = numpy.r_[_grad].reshape(grad.shape)
        _err = reldiff(grad, _grad)
        assert _err<check_eps, 'Fail grad of bottom|%s, err|%g > %g\n%s%s%s'%(
            iG, _err, check_eps, _pp('grad', grad), _pp('_grad',_grad) , _pp('diff',grad-_grad) 
        )
        
        logging.debug('pass blobs|%s ... '%iG)







