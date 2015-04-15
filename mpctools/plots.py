import numpy as np
import matplotlib.pyplot as plt

"""
Contains all of the plotting functions for mpc-tools-casadi.
"""

def mpcplot(x,u,t,xsp=None,fig=None,xinds=None,uinds=None,tightness=.5,
            title=None,timefirst=True):
    """
    Makes a plot of the state and control trajectories for an mpc problem.
    
    Inputs x and u should be n by N+1 and p by N numpy arrays. xsp if provided
    should be the same size as x. t should be a numpy N+1 vector.
    
    If given, fig is the matplotlib figure handle to plot everything. If not
    given, a new figure is used.
    
    xinds and uinds are optional lists of indices to plot. If not given, all
    indices of x and u are plotted.
    
    Returns the figure handle used for plotting.
    """
    # Transpose data if time is the first dimension; we need it second.
    if timefirst:
        x = x.T
        u = u.T
    
    # Process arguments.
    if xinds is None:
        xinds = range(x.shape[0])
    if uinds is None:
        uinds = range(u.shape[0])
    if fig is None:
        fig = plt.figure()
    if xsp is None:
        xlspec = "-k"
        ulspec = "-k"
        plotxsp = False
    else:
        xlspec = "-g"
        ulspec = "-b"
        plotxsp = True
    
    # Figure out how many plots to make.
    numrows = max(len(xinds),len(uinds))
    if numrows == 0: # No plots to make.
        return None
    numcols = 2
    
    # u plots.
    u = np.hstack((u,u[:,-1:])) # Repeat last element for stairstep plot.
    for i in range(len(uinds)):
        uind = uinds[i]
        a = fig.add_subplot(numrows,numcols,numcols*(i+1))
        a.step(t,np.squeeze(u[uind,:]),ulspec)
        a.set_xlabel("Time")
        a.set_ylabel("Control %d" % (uind + 1))
        zoomaxis(a,yscale=1.05)
    
    # x plots.    
    for i in range(len(xinds)):
        xind = xinds[i]
        a = fig.add_subplot(numrows,numcols,numcols*(i+1) - 1)
        a.hold("on")
        a.plot(t,np.squeeze(x[xind,:]),xlspec,label="System")
        if plotxsp:
            a.plot(t,np.squeeze(xsp[xind,:]),"--r",label="Setpoint")
            plt.legend(loc="best")
        a.set_xlabel("Time")
        a.set_ylabel("State %d" % (xind + 1))
        zoomaxis(a,yscale=1.05)
    
    # Layout tightness.
    if not tightness is None:
        fig.tight_layout(pad=tightness)
    if title is not None:
        fig.canvas.set_window_title(title)       
    
    return fig


def zoomaxis(axes=None,xscale=None,yscale=None):
    """
    Zooms the axes by a specified amounts (positive multipliers).
    
    If axes is None, plt.gca() is used.
    """
    # Grab default axes if necessary.
    if axes is None:
        axes = plt.gca()
    
    # Make sure input is valid.
    if (xscale is not None and xscale <= 0) or (yscale is not None and yscale <= 0):
        raise ValueError("Scale values must be strictly positive.")
    
    # Adjust axes limits.
    for (scale,getter,setter) in [(xscale,axes.get_xlim,axes.set_xlim), (yscale,axes.get_ylim,axes.set_ylim)]:
        if scale is not None:
            # Subtract one from each because of how we will calculate things.            
            scale -= 1
   
            # Get limits and change them.
            (minlim,maxlim) = getter()
            offset = .5*scale*(maxlim - minlim)
            setter(minlim - offset, maxlim + offset)