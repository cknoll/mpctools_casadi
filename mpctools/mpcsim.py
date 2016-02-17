# mpcsim is a graphical user interface for the mpc-tools-casadi package
#
# Tom Badgwell June 2015
#
# to do list:
#
# - implement option to disable open-loop predictions while closed loop
# - implement file open/closed options (quit the current window)
# - disable the menu options depending on mode
# - implement a menu for the unmeasured disturbances - tun factors
# - implement plots for the unmeasured disturbance variables
# - implement option to show process diagram
# - implement re-initialize option
#

import Tkinter as tk
import tkMessageBox as tkmsg
from   tkFileDialog import askopenfilename
from   tkSimpleDialog import askfloat
from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt

def makegui(simcon):

    # create main window
    root = tk.Tk()
    root.title('MPC-Sim')
    simcon.root = root

    # create the menus
    menubar = makemenus(root, simcon)

    # create the run panel on the menubar
    rpanel = RunPanel(menubar)

    # create the control panel on the menubar
    cpanel = ConPanel(menubar)
    
    # Add the step button and the reset button.
    stepbutton = tk.Button(menubar)
    stepbutton.configure(text="Manual\nStep")
    stepbutton.pack(side=tk.LEFT)
    
    resetbutton = tk.Button(menubar)
    resetbutton.configure(text="Reset\nSimulation")
    resetbutton.pack(side=tk.LEFT)
    
    # add the simulation name box
    makename(menubar, simcon.simname)    
    
    # fill in remaining space on the menubar
    fillspace(menubar)

    # create the trend plots
    trndplt = Trndplt(root, simcon, rpanel, cpanel, stepbutton, resetbutton)
    trndplt.autosim()

    # start the main loop
    root.mainloop()

def notdone():
    tkmsg.showerror('Not implemented', 'Not yet available')

def my_add_command(mymenu, var, desc):
    mymenu.add_command(label='Set ' + desc, 
    command=lambda: setvalue(var, desc), 
    underline=0)

def openfile(simcon):
    f = askopenfilename()
    simcon.root.destroy()
    execfile(f)

def setvalue(var, desc):

    if desc == 'Value':

        value = float(var.value)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext)
#        if value != None: var.value  = value*np.ones((1,1))
        if value != None: var.value  = value

    elif desc == 'SS Target':

        value = float(var.sstarg)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext)
        if value != None: 
            var.sstarg = value
            var.chflag = 1

    elif desc == 'SS R Weight':

        value = float(var.ssrval)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: 
            var.ssrval = value
            var.chflag = 1

    elif desc == 'SS Q Weight':

        value = float(var.ssqval)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: 
            var.ssval = value
            var.chflag = 1

    elif desc == 'Target':

        value = float(var.target)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext)
        if value != None: 
            var.target = value
#            var.chflag = 1

    elif desc == 'Setpoint':

        value = float(var.setpoint)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext)
        if value != None: 
            var.setpoint = value
#            var.chflag = 1

    elif desc == 'Q Weight':

        value = float(var.qvalue)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: 
            var.qvalue = value
            var.chflag = 1

    elif desc == 'R Weight':

        value = float(var.rvalue)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: 
            var.rvalue = value
            var.chflag = 1

    elif desc == 'S Weight':

        value = float(var.svalue)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: 
            var.svalue = value
            var.chflag = 1

    elif desc == 'Max Limit':

        value = float(var.maxlim)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=var.minlim)
        if value != None: 
            var.maxlim = value
            var.chflag = 1

    elif desc == 'Min Limit':

        value = float(var.minlim)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, maxvalue=var.maxlim)
        if value != None: 
            var.minlim = value
            var.chflag = 1

    elif desc == 'ROC Limit':

        value = float(var.roclim)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: 
            var.roclim = value
            var.chflag = 1

    elif desc == 'Plot High Limit':

        value = float(var.pltmax)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=var.pltmin)
        if value != None: var.pltmax = value

    elif desc == 'Plot Low Limit':

        value = float(var.pltmin)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, maxvalue=var.pltmax)
        if value != None: var.pltmin = value

    elif desc == 'Process Noise':

        value = float(var.noise)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: var.noise = value

    elif desc == 'Model Noise':

        value = float(var.mnoise)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: 
            var.mnoise = value
            var.chflag = 1

    elif desc == 'Process Step Dist.':

        value = float(var.dist)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext)
        if value != None: var.dist = value

    elif desc == 'Refresh Int':

        value = float(var.refint)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=10.0, maxvalue=1000.0)
        if value != None: var.refint = value

    elif desc == 'Noise Factor':

        value = float(var.value)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: var.value = value

    elif desc == 'A Value':

        value = float(var.value)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=-10.0, maxvalue=10.0)
        if value != None: 
            var.value = value
            var.chflag = 1

    elif desc == 'Gain Mismatch Factor':

        value = float(var.gfac)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0)
        if value != None: 
            var.gfac = value
            var.chflag = 1

    elif desc == 'Disturbance Model':

        value = float(var.value)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=1.0, maxvalue=5.0)
        if value != None: 
            var.value   = value
            var.chflag = 1

    elif desc == 'Control Gain':

        value = float(var.value)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=-10.0, maxvalue=10.0)
        if value != None: 
            var.value   = value
            var.chflag = 1

    elif desc == 'Reset Time':

        value = float(var.value)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0, maxvalue=10000.0)
        if value != None: 
            var.value   = value
            var.chflag = 1

    elif desc == 'Derivative Time':

        value = float(var.value)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=0.0, maxvalue=100.0)
        if value != None: 
            var.value   = value
            var.chflag = 1

    elif desc == 'Heat Of Reaction':

        value = float(var.value)
        entrytext = desc + ' currently ' + str(value) + ', enter new value'
        value = askfloat(var.name, entrytext, minvalue=-1e6, maxvalue=1e6)
        if value != None: 
            var.value   = value
            var.chflag = 1

    else:

        notdone()

def showhelp():
    tkmsg.showinfo('About MPC-Sim','MPC-Sim is a GUI for the '
                   'mpc-tools-casadi package (Tom Badgwell)')

def makemenus(win, simcon):

    mvlist = simcon.mvlist
    dvlist = simcon.dvlist
    cvlist = simcon.cvlist
    xvlist = simcon.xvlist
    oplist = simcon.oplist

    menubar = tk.Frame(win)
    menubar.config(bd=2, relief=tk.GROOVE)
    menubar.pack(side=tk.TOP, fill=tk.X)

    # build the file menu

    fbutton = tk.Menubutton(menubar, text='File', underline=0)
    fbutton.pack(side=tk.LEFT)
    filemenu = tk.Menu(fbutton, tearoff=0)
#    filemenu.add_command(label='Open',  command=lambda: openfile(simcon),  underline=0)
#    filemenu.add_command(label='Close', command=notdone,  underline=0)
    filemenu.add_command(label='Exit',  command=win.quit, underline=0)
    fbutton.config(menu=filemenu)

    # build the MV menu

    mbutton = tk.Menubutton(menubar, text='MVs', underline=0)
    mbutton.pack(side=tk.LEFT)
    mvsmenu = tk.Menu(mbutton, tearoff=0)
    mbutton.config(menu=mvsmenu)

    for mv in mvlist:

        mvmenu = tk.Menu(mvsmenu, tearoff=False)
        if ("value"  in mv.menu): my_add_command(mvmenu, mv, 'Value')
        if ("sstarg" in mv.menu): my_add_command(mvmenu, mv, 'SS Target') 
        if ("ssrval" in mv.menu): my_add_command(mvmenu, mv, 'SS R Weight') 
        if ("target" in mv.menu): my_add_command(mvmenu, mv, 'Target') 
        if ("rvalue" in mv.menu): my_add_command(mvmenu, mv, 'R Weight') 
        if ("svalue" in mv.menu): my_add_command(mvmenu, mv, 'S Weight') 
        if ("maxlim" in mv.menu): my_add_command(mvmenu, mv, 'Max Limit') 
        if ("minlim" in mv.menu): my_add_command(mvmenu, mv, 'Min Limit') 
        if ("roclim" in mv.menu): my_add_command(mvmenu, mv, 'ROC Limit') 
        if ("pltmax" in mv.menu): my_add_command(mvmenu, mv, 'Plot High Limit') 
        if ("pltmin" in mv.menu): my_add_command(mvmenu, mv, 'Plot Low Limit') 
        if ("noise"  in mv.menu): my_add_command(mvmenu, mv, 'Noise') 
        if ("dist"   in mv.menu): my_add_command(mvmenu, mv, 'Step Disturbance') 
        mvsmenu.add_cascade(label=mv.name, menu=mvmenu, underline = 0)

    # build the DV menu if there are DVs

    if (len(dvlist) > 0):

        dbutton = tk.Menubutton(menubar, text='DVs', underline=0)
        dbutton.pack(side=tk.LEFT)
        dvsmenu = tk.Menu(dbutton, tearoff=0)
        dbutton.config(menu=dvsmenu)

        for dv in dvlist:

            dvmenu = tk.Menu(dvsmenu, tearoff=False)
            if ("value"  in dv.menu): my_add_command(dvmenu, dv, 'Value')
            if ("pltmax" in dv.menu): my_add_command(dvmenu, dv, 'Plot High Limit') 
            if ("pltmin" in dv.menu): my_add_command(dvmenu, dv, 'Plot Low Limit') 
            if ("noise"  in dv.menu): my_add_command(dvmenu, dv, 'Process Noise') 
            dvsmenu.add_cascade(label=dv.name, menu=dvmenu, underline = 0)

    # build the XV menu if there are XVs

    if (len(xvlist) > 0):

        xbutton = tk.Menubutton(menubar, text='XVs', underline=0)
        xbutton.pack(side=tk.LEFT)
        xvsmenu = tk.Menu(xbutton, tearoff=0)
        xbutton.config(menu=xvsmenu)

        for xv in xvlist:

            xvmenu = tk.Menu(xvsmenu, tearoff=False)
            if ("value"    in xv.menu): my_add_command(xvmenu, xv, 'Value')
            if ("sstarg"   in xv.menu): my_add_command(xvmenu, xv, 'SS Target') 
            if ("ssqval"   in xv.menu): my_add_command(xvmenu, xv, 'SS Q Weight') 
            if ("setpoint" in xv.menu): my_add_command(xvmenu, xv, 'Setpoint') 
            if ("qvalue"   in xv.menu): my_add_command(xvmenu, xv, 'Q Weight') 
            if ("maxlim"   in xv.menu): my_add_command(xvmenu, xv, 'Max Limit') 
            if ("minlim"   in xv.menu): my_add_command(xvmenu, xv, 'Min Limit') 
            if ("pltmax"   in xv.menu): my_add_command(xvmenu, xv, 'Plot High Limit') 
            if ("pltmin"   in xv.menu): my_add_command(xvmenu, xv, 'Plot Low Limit') 
            if ("mnoise"   in xv.menu): my_add_command(xvmenu, xv, 'Model Noise') 
            if ("noise"    in xv.menu): my_add_command(xvmenu, xv, 'Process Noise') 
            if ("dist"     in xv.menu): my_add_command(xvmenu, xv, 'Process Step Dist.') 
            xvsmenu.add_cascade(label=xv.name, menu=xvmenu, underline = 0)

    # build the CV menu

    cbutton = tk.Menubutton(menubar, text='CVs', underline=0)
    cbutton.pack(side=tk.LEFT)
    cvsmenu = tk.Menu(cbutton, tearoff=0)
    cbutton.config(menu=cvsmenu)

    for cv in cvlist:

        cvmenu = tk.Menu(cvsmenu, tearoff=False)
        if ("value"    in cv.menu): my_add_command(cvmenu, cv, 'Value')
        if ("sstarg"   in cv.menu): my_add_command(cvmenu, cv, 'SS Target') 
        if ("ssqval"   in cv.menu): my_add_command(cvmenu, cv, 'SS Q Weight') 
        if ("setpoint" in cv.menu): my_add_command(cvmenu, cv, 'Setpoint') 
        if ("qvalue"   in cv.menu): my_add_command(cvmenu, cv, 'Q Weight') 
        if ("maxlim"   in cv.menu): my_add_command(cvmenu, cv, 'Max Limit') 
        if ("minlim"   in cv.menu): my_add_command(cvmenu, cv, 'Min Limit') 
        if ("pltmax"   in cv.menu): my_add_command(cvmenu, cv, 'Plot High Limit') 
        if ("pltmin"   in cv.menu): my_add_command(cvmenu, cv, 'Plot Low Limit') 
        if ("mnoise"   in cv.menu): my_add_command(cvmenu, cv, 'Model Noise') 
        if ("noise"    in cv.menu): my_add_command(cvmenu, cv, 'Process Noise') 
        if ("dist"     in cv.menu): my_add_command(cvmenu, cv, 'Process Step Dist.') 
        cvsmenu.add_cascade(label=cv.name, menu=cvmenu, underline = 0)

    # build the options menu

    obutton = tk.Menubutton(menubar, text='Options', underline=0)
    obutton.pack(side=tk.LEFT)
    opsmenu = tk.Menu(obutton, tearoff=0)
    obutton.config(menu=opsmenu)

    for op in oplist:

        opmenu = tk.Menu(opsmenu, tearoff=False)
        my_add_command(opmenu, op, op.desc) 
        opsmenu.add_cascade(label=op.name, menu=opmenu, underline = 0)

    # build the help menu

    hbutton = tk.Menubutton(menubar, text='Help', underline=0)
    hbutton.pack(side=tk.LEFT)
    helpmenu = tk.Menu(hbutton, tearoff=0)
    helpmenu.add_command(label='About', command=showhelp,  underline=0)
    hbutton.config(menu=helpmenu)

    return menubar

class Trndplt:

    def __init__(self, parent, simcon, runpause, opnclsd, stepbutton,
                 resetbutton):

        # store inputs

        self.N        = simcon.N
        self.Nm1      = simcon.N-1
        self.deltat   = simcon.deltat
        self.simcon   = simcon
        self.mvlist   = simcon.mvlist
        self.dvlist   = simcon.dvlist
        self.cvlist   = simcon.cvlist
        self.xvlist   = simcon.xvlist
        self.oplist   = simcon.oplist
        self.runpause = runpause
        self.opnclsd  = opnclsd
        self.stepbutton = stepbutton
        self.resetbutton = resetbutton
        self.refint   = simcon.refint
        self.runsim   = simcon.runsim
        self.k        = 0
        self.xvec     = np.arange(-simcon.N,0)
        self.parent = parent # Save this guy.

        # build the figure

        self.fig      = plt.Figure()

        # determine the subplot dimensions

        self.nmvs      = len(self.mvlist)
        self.ndvs      = len(self.dvlist)
        self.ncvs      = len(self.cvlist)
        self.nxvs      = len(self.xvlist)
        self.ninputs   = self.nmvs + self.ndvs
        self.noutputs  = self.ncvs
        self.nrows     = max(self.ninputs,self.nxvs,self.noutputs)
        self.ncols     = 2
        if (self.nxvs > 0): self.ncols = 3
        self.submat    = str(self.nrows) + str(self.ncols)

        # initialize values

        self.mvaxes    = []
        self.dvaxes    = []
        self.cvaxes    = []
        self.xvaxes    = []

        self.mvlines   = []
        self.mvmxlines = []
        self.mvmnlines = []

        self.dvlines   = []

        self.cvlines   = []
        self.cveslines = []
        self.cvsplines = []
        self.cvmxlines = []
        self.cvmnlines = []

        self.xvlines   = []
        self.xveslines = []
        self.xvsplines = []
        self.xvmxlines = []
        self.xvmnlines = []

        self.fomvlines  = []
        self.fcmvlines  = []
        self.fmvmxlines = []
        self.fmvmnlines = []

        self.fdvlines   = []

        self.focvlines  = []
        self.fccvlines  = []
        self.fcvsplines = []
        self.fcvmxlines = []
        self.fcvmnlines = []

        self.foxvlines  = []
        self.fcxvlines  = []
        self.fxvsplines = []
        self.fxvmxlines = []
        self.fxvmnlines = []

        # set up the subplots

        imv = 1;
        idv = 1;
        icv = 1;
        ixv = 1;

        for irow in range(1,self.nrows+1):

            isub = (irow - 1)*self.ncols + 1

            if (imv <= self.nmvs):
                mv = self.mvlist[imv-1]
                mvaxis = self.fig.add_subplot(self.nrows,self.ncols,isub,
                         ylabel=mv.name+' '+mv.units, 
                         title=mv.desc,
                         ylim=(mv.pltmin,mv.pltmax))
                mvaxis.grid(True,'major')
                self.mvaxes.append(mvaxis)
                imv += 1
        
            if (idv <= self.ndvs) and (irow > self.nmvs):
                dv = self.dvlist[idv-1]
                dvaxis = self.fig.add_subplot(self.nrows,self.ncols,isub,
                         ylabel=dv.name+' '+dv.units, 
                         title=dv.desc,
                         ylim=(dv.pltmin,dv.pltmax))
                dvaxis.grid(True,'major')
                self.dvaxes.append(dvaxis)
                idv +=1

            if (ixv <= self.nxvs):
                isub += 1
                xv = self.xvlist[ixv-1]
                xvaxis = self.fig.add_subplot(self.nrows,self.ncols,isub,
                         ylabel=xv.name+' '+xv.units,
                         title=xv.desc,
                         ylim=(xv.pltmin,xv.pltmax))
                xvaxis.grid(True,'major')
                self.xvaxes.append(xvaxis)
                ixv +=1

            if (icv <= self.ncvs):
                isub += 1
                cv = self.cvlist[icv-1]
                cvaxis = self.fig.add_subplot(self.nrows,self.ncols,isub,
                         ylabel=cv.name+' '+cv.units, 
                         title=cv.desc,
                         ylim=(cv.pltmin,cv.pltmax))
                cvaxis.grid(True,'major')
                self.cvaxes.append(cvaxis)
                icv += 1

        # plot initial values

        for mv in self.mvlist:

            mvndx   = self.mvlist.index(mv)
            mvaxis  = self.mvaxes[mvndx]

            yvec    = mv.value*np.ones((self.N,1))
            mvline, = mvaxis.plot(self.xvec, yvec, 'k')
            self.mvlines.append(mvline)

            yvec    = mv.maxlim*np.ones((self.N,1))
            mvmxline, = mvaxis.plot(self.xvec, yvec, 'r', ls='--')
            self.mvmxlines.append(mvmxline)

            yvec    = mv.minlim*np.ones((self.N,1))
            mvmnline, = mvaxis.plot(self.xvec, yvec, 'r', ls='--')
            self.mvmnlines.append(mvmnline)

            yvec      = mv.value*np.ones((mv.Nf,1))
            xvec      = np.arange(0,mv.Nf)
            fomvline, = mvaxis.plot(xvec, yvec, 'b', ls='--')
            fcmvline, = mvaxis.plot(xvec, yvec, 'b')
            self.fomvlines.append(fomvline)
            self.fcmvlines.append(fcmvline)

            yvec       = mv.maxlim*np.ones((mv.Nf,1))
            fmvmxline, = mvaxis.plot(xvec, yvec, 'r', ls='--')
            self.fmvmxlines.append(fmvmxline)

            yvec       = mv.minlim*np.ones((mv.Nf,1))
            fmvmnline, = mvaxis.plot(xvec, yvec, 'r', ls='--')
            self.fmvmnlines.append(fmvmnline)

            mvaxis.plot((0,0),(-1e6,1e6), 'r')

        for dv in self.dvlist:

            dvndx   = self.dvlist.index(dv)
            dvaxis  = self.dvaxes[dvndx]

            yvec    = dv.value*np.ones((self.N,1))
            dvline, = dvaxis.plot(self.xvec, yvec, 'k')
            self.dvlines.append(dvline)

            yvec      = dv.value*np.ones((dv.Nf,1))
            xvec      = np.arange(0,dv.Nf)
            fdvline, = dvaxis.plot(xvec, yvec, 'b')
            self.fdvlines.append(fdvline)

            dvaxis.plot((0,0),(-1e6,1e6), 'r')

        for xv in self.xvlist:

            xvndx   = self.xvlist.index(xv)
            xvaxis  = self.xvaxes[xvndx]

            yvec    = xv.value*np.ones((self.N,1))
            xvline, = xvaxis.plot(self.xvec, yvec, 'k')
            self.xvlines.append(xvline)

            yvec    = xv.est*np.ones((self.N,1))
            xvesline, = xvaxis.plot(self.xvec, yvec, 'b')
            self.xveslines.append(xvesline)

#            yvec    = xv.setpoint*np.ones((self.N,1))
#            xvspline, = xvaxis.plot(self.xvec, yvec, 'g')
#            self.xvsplines.append(xvspline)

#            yvec    = xv.maxlim*np.ones((self.N,1))
#            xvmxline, = xvaxis.plot(self.xvec, yvec, 'r', ls='--')
#            self.xvmxlines.append(xvmxline)

#            yvec    = xv.minlim*np.ones((self.N,1))
#            xvmnline, = xvaxis.plot(self.xvec, yvec, 'r', ls='--')
#            self.xvmnlines.append(xvmnline)

            yvec      = xv.value*np.ones((xv.Nf,1))
            xvec      = np.arange(0,xv.Nf)
            foxvline, = xvaxis.plot(xvec, yvec, 'b', ls='--')
            fcxvline, = xvaxis.plot(xvec, yvec, 'b')
            self.foxvlines.append(foxvline)
            self.fcxvlines.append(fcxvline)

#            yvec       = xv.setpoint*np.ones((xv.Nf,1))
#            fxvspline, = xvaxis.plot(xvec, yvec, 'g')
#            self.fxvsplines.append(fxvspline)

#            yvec       = xv.maxlim*np.ones((xv.Nf,1))
#            fxvmxline, = xvaxis.plot(xvec, yvec, 'r', ls='--')
#            self.fxvmxlines.append(fxvmxline)

#            yvec       = xv.minlim*np.ones((xv.Nf,1))
#            fxvmnline, = xvaxis.plot(xvec, yvec, 'r', ls='--')
#            self.fxvmnlines.append(fxvmnline)

            xvaxis.plot((0,0),(-1e6,1e6), 'r')

        for cv in self.cvlist:

            cvndx   = self.cvlist.index(cv)
            cvaxis  = self.cvaxes[cvndx]

            yvec    = cv.value*np.ones((self.N,1))
            cvline, = cvaxis.plot(self.xvec, yvec, 'k')
            self.cvlines.append(cvline)

            yvec    = cv.est*np.ones((self.N,1))
            cvesline, = cvaxis.plot(self.xvec, yvec, 'b')
            self.cveslines.append(cvesline)

            yvec    = cv.setpoint*np.ones((self.N,1))
            cvspline, = cvaxis.plot(self.xvec, yvec, 'g')
            self.cvsplines.append(cvspline)

            yvec    = cv.maxlim*np.ones((self.N,1))
            cvmxline, = cvaxis.plot(self.xvec, yvec, 'r', ls='--')
            self.cvmxlines.append(cvmxline)

            yvec    = cv.minlim*np.ones((self.N,1))
            cvmnline, = cvaxis.plot(self.xvec, yvec, 'r', ls='--')
            self.cvmnlines.append(cvmnline)

            yvec      = cv.value*np.ones((cv.Nf,1))
            xvec      = np.arange(0,cv.Nf)
            focvline, = cvaxis.plot(xvec, yvec, 'b', ls='--')
            fccvline, = cvaxis.plot(xvec, yvec, 'b')
            self.focvlines.append(focvline)
            self.fccvlines.append(fccvline)

            yvec       = cv.setpoint*np.ones((cv.Nf,1))
            fcvspline, = cvaxis.plot(xvec, yvec, 'g')
            self.fcvsplines.append(fcvspline)

            yvec       = cv.maxlim*np.ones((cv.Nf,1))
            fcvmxline, = cvaxis.plot(xvec, yvec, 'r', ls='--')
            self.fcvmxlines.append(fcvmxline)

            yvec       = cv.minlim*np.ones((cv.Nf,1))
            fcvmnline, = cvaxis.plot(xvec, yvec, 'r', ls='--')
            self.fcvmnlines.append(fcvmnline)

            cvaxis.plot((0,0),(-1e6,1e6), 'r')

        # attach figure to parent and configure buttons.

        self.canvas   = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
        
        self.stepbutton.configure(command=self.simulate)
        self.stepbutton.focus_force()

        self.resetbutton.configure(command=self.reset)

    def autosim(self, repeat=True):
        """Loop to automatically run simulation."""
        if self.runpause.status.get() == 1:
            self.simulate()
        if repeat:
            self.parent.after(int(self.refint), self.autosim)
    
    def simulate(self, i=0):
        """Run one simulation step if not paused."""
        # Run step.
        self.simcon.runsim(self.k, self.simcon, self.opnclsd)

        # update the trends
        self.pltvals()

        # increment the iteration count
        self.k += 1

    def reset(self):
        """Resets simulation to initial values."""
        self.k = 0
        #TODO: rest all plot data and options.

    def pltvals(self, updatefig=True):

        # update mv trends

        for mv in self.mvlist:

            mvndx = self.mvlist.index(mv)
            mvaxis = self.mvaxes[mvndx]

            mvaxis.set_ylim([mv.pltmin, mv.pltmax])
#            mvaxis.plot((0,0),(mv.pltmin,mv.pltmax), 'r')

            mvline = self.mvlines[mvndx]
            ydata  = mvline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = mv.value
            mvline.set_ydata(ydata)

            mvmxline = self.mvmxlines[mvndx]
            ydata  = mvmxline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = mv.maxlim
            mvmxline.set_ydata(ydata)

            mvmnline = self.mvmnlines[mvndx]
            ydata  = mvmnline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = mv.minlim
            mvmnline.set_ydata(ydata)

            fomvline = self.fomvlines[mvndx]
            fomvline.set_ydata(mv.olpred)

            fcmvline = self.fcmvlines[mvndx]
            fcmvline.set_ydata(mv.clpred)

            fmvmxline  = self.fmvmxlines[mvndx]
            yvec       = mv.maxlim*np.ones((mv.Nf,1))
            fmvmxline.set_ydata(yvec)

            fmvmnline  = self.fmvmnlines[mvndx]
            yvec       = mv.minlim*np.ones((mv.Nf,1))
            fmvmnline.set_ydata(yvec)

        # update dv trends

        for dv in self.dvlist:

            dvndx = self.dvlist.index(dv)
            dvaxis = self.dvaxes[dvndx]

            dvaxis.set_ylim([dv.pltmin, dv.pltmax])
#            dvaxis.plot((0,0),(dv.pltmin,dv.pltmax), 'r')

            dvline = self.dvlines[dvndx]
            ydata  = dvline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = dv.value
            dvline.set_ydata(ydata)

            fdvline = self.fdvlines[dvndx]
            yvec    = dv.value*np.ones((dv.Nf,1))
            fdvline.set_ydata(yvec)

        # update xv trends

        for xv in self.xvlist:

            xvndx = self.xvlist.index(xv)
            xvaxis = self.xvaxes[xvndx]

            xvaxis.set_ylim([xv.pltmin, xv.pltmax])
#            xvaxis.plot((0,0),(xv.pltmin,xv.pltmax), 'r')

            xvline = self.xvlines[xvndx]
            ydata  = xvline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = xv.value
            xvline.set_ydata(ydata)

            xvesline = self.xveslines[xvndx]
            ydata  = xvesline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = xv.est
            xvesline.set_ydata(ydata)

#            xvspline = self.xvsplines[xvndx]
#            ydata  = xvspline.get_ydata()
#            ydata  = np.roll(ydata,-1,0)
#            ydata[self.Nm1] = xv.setpoint
#            xvspline.set_ydata(ydata)

#            xvmxline = self.xvmxlines[xvndx]
#            ydata  = xvmxline.get_ydata()
#            ydata  = np.roll(ydata,-1,0)
#            ydata[self.Nm1] = xv.maxlim
#            xvmxline.set_ydata(ydata)

#            xvmnline = self.xvmnlines[xvndx]
#            ydata  = xvmnline.get_ydata()
#            ydata  = np.roll(ydata,-1,0)
#            ydata[self.Nm1] = xv.minlim
#            xvmnline.set_ydata(ydata)

            foxvline = self.foxvlines[xvndx]
            foxvline.set_ydata(xv.olpred)

            fcxvline = self.fcxvlines[xvndx]
            fcxvline.set_ydata(xv.clpred)

#            fxvspline  = self.fxvsplines[xvndx]
#            yvec       = xv.setpoint*np.ones((xv.Nf,1))
#            fxvspline.set_ydata(yvec)

#            fxvmxline  = self.fxvmxlines[xvndx]
#            yvec       = xv.maxlim*np.ones((xv.Nf,1))
#            fxvmxline.set_ydata(yvec)

#            fxvmnline  = self.fxvmnlines[xvndx]
#            yvec       = xv.minlim*np.ones((xv.Nf,1))
#            fxvmnline.set_ydata(yvec)

        # update cv trends

        for cv in self.cvlist:

            cvndx = self.cvlist.index(cv)
            cvaxis = self.cvaxes[cvndx]

            cvaxis.set_ylim([cv.pltmin, cv.pltmax])
#            cvaxis.plot((0,0),(cv.pltmin,cv.pltmax), 'r')

            cvline = self.cvlines[cvndx]
            ydata  = cvline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = cv.value
            cvline.set_ydata(ydata)

            cvesline = self.cveslines[cvndx]
            ydata  = cvesline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = cv.est
            cvesline.set_ydata(ydata)

            cvspline = self.cvsplines[cvndx]
            ydata  = cvspline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = cv.setpoint
            cvspline.set_ydata(ydata)

            cvmxline = self.cvmxlines[cvndx]
            ydata  = cvmxline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = cv.maxlim
            cvmxline.set_ydata(ydata)

            cvmnline = self.cvmnlines[cvndx]
            ydata  = cvmnline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.Nm1] = cv.minlim
            cvmnline.set_ydata(ydata)

            focvline = self.focvlines[cvndx]
            focvline.set_ydata(cv.olpred)

            fccvline = self.fccvlines[cvndx]
            fccvline.set_ydata(cv.clpred)

            fcvspline  = self.fcvsplines[cvndx]
            yvec       = cv.setpoint*np.ones((cv.Nf,1))
            fcvspline.set_ydata(yvec)

            fcvmxline  = self.fcvmxlines[cvndx]
            yvec       = cv.maxlim*np.ones((cv.Nf,1))
            fcvmxline.set_ydata(yvec)

            fcvmnline  = self.fcvmnlines[cvndx]
            yvec       = cv.minlim*np.ones((cv.Nf,1))
            fcvmnline.set_ydata(yvec)
            
        # Update figure.
        if updatefig:
            self.fig.canvas.draw()

class RadioPanel(object):
    def __init__(self, parent, title="", lbutton="", rbutton=""):
        self.status = tk.IntVar()
        self.frame = tk.Frame(parent)
        self.frame.config(bd=2, relief=tk.GROOVE)
        self.frame.pack(side=tk.LEFT)
        msg = tk.Label(self.frame, text=title)
        msg.pack(side=tk.TOP)
        pauseb = tk.Radiobutton(self.frame, text=lbutton, command=self.setbg, 
                                variable=self.status, value=0)
        pauseb.pack(side=tk.LEFT)
        runb = tk.Radiobutton(self.frame, text=rbutton, command=self.setbg,
                              variable=self.status, value=1)
        runb.pack(side=tk.LEFT)
        self.status.set(0)
        self.frame.config(bg='red')

    def setbg(self):
        if self.status.get() == 0:
            self.frame.config(bg='red')
        if self.status.get() == 1:
            self.frame.config(bg='green')

class RunPanel(RadioPanel):
    def __init__(self, parent):
        super(RunPanel, self).__init__(parent, title="Sim Status",
                                       lbutton="Pause", rbutton="Run")    
    @property
    def rframe(self):
        return self.frame

class ConPanel(RadioPanel):
    def __init__(self, parent):
        super(ConPanel, self).__init__(parent, title="Loop Status",
                                       lbutton="Open", rbutton="Closed")    
    @property
    def rframe(self):
        return self.frame

class Option:

    def __init__(self, name=' ', desc=' ', value=0.0):

        self.name   = name
        self.desc   = desc
        self.value  = value
        self.chflag = 0

def makename(parent, simname):

    nameframe = tk.Frame(parent)
    nameframe.config(bd=2, relief=tk.GROOVE)
    nameframe.pack(side=tk.LEFT)
    padname = ' ' + simname + ' '
    namebox = tk.Label(nameframe, text=padname, font=('ariel', 12, 'bold'))
    namebox.pack(side=tk.LEFT)

def fillspace(parent):

    fillframe = tk.Frame(parent)
    fillframe.config(bg='blue')
    fillframe.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)

class MVobj:

    nmvs = 0;

    def __init__(self, name=' ', desc=' ', units= ' ',
                 value=0.0*np.ones((1,1)),
                 sstarg=0.0, ssrval=0.01,
                 target=0.0, rvalue=0.001, svalue=0.001, 
                 maxlim=1.0e10, minlim=-1.0e10, roclim=1.0e10,
                 pltmax=100.0, pltmin=0.0,
                 noise=0.0, dist=0.0, Nf=0,
                 menu=["value","sstarg","ssrval","target","rvalue","svalue",
                       "maxlim","minlim","roclim","pltmax","pltmin","noise",
                       "dist"]):
        
        MVobj.nmvs += 1

        self.name   = name
        self.desc   = desc
        self.units  = units
        self.value  = value
        self.est    = value
        self.sstarg = sstarg
        self.ssrval = ssrval
        self.target = target
        self.rvalue = rvalue
        self.svalue = svalue
        self.maxlim = maxlim
        self.minlim = minlim
        self.roclim = roclim
        self.pltmax = pltmax
        self.pltmin = pltmin
        self.noise  = noise
        self.dist   = dist
        self.ref    = value
        self.chflag = 0
        self.Nf     = Nf
        self.olpred = value*np.ones((Nf,1))
        self.clpred = value*np.ones((Nf,1))
        self.menu   = menu

class DVobj:

    ndvs = 0;

    def __init__(self, name=' ', desc=' ', units=' ', 
                 value=0.0*np.ones((1,1)),
                 pltmax=100.0, pltmin=0.0,
                 noise=0.0, Nf=0,
                 menu=["value","pltmax","pltmin","noise"]):
        
        DVobj.ndvs += 1

        self.name   = name
        self.desc   = desc
        self.units  = units
        self.value  = value
        self.est    = value
        self.pltmax = pltmax
        self.pltmin = pltmin
        self.noise  = noise
        self.ref    = value
        self.chflag = 0
        self.Nf     = Nf
        self.olpred = value*np.ones((Nf,1))
        self.clpred = value*np.ones((Nf,1))
        self.menu   = menu

class CVobj:

    ncvs = 0;

    def __init__(self, name=' ', desc=' ', units=' ', 
                 value=0.0*np.ones((1,1)),
                 sstarg=0.0, ssqval=1.0,
                 setpoint=0.0, qvalue=1.0,
                 maxlim=1.0e10, minlim=-1.0e10, roclim=1.0e10,
                 pltmax=100.0, pltmin=0.0,
                 noise=0.0, mnoise=0.000001, dist=0.0, Nf=0, bias=0.0,
                 menu=["value","sstarg","ssqval","setpoint","qvalue",
                       "maxlim","minlim","pltmax","pltmin","mnoise",
                       "noise","dist"]):
        
        CVobj.ncvs += 1

        self.name   = name
        self.desc   = desc
        self.units  = units
        self.value  = value
        self.est    = value
        self.sstarg = sstarg
        self.ssqval = ssqval
        self.setpoint = setpoint
        self.qvalue = qvalue
        self.maxlim = maxlim
        self.minlim = minlim
        self.pltmax = pltmax
        self.pltmin = pltmin
        self.noise  = noise
        self.mnoise = mnoise
        self.dist   = dist
        self.ref    = value
        self.chflag = 0
        self.Nf     = Nf
        self.bias   = bias
        self.olpred = value*np.ones((Nf,1))
        self.clpred = value*np.ones((Nf,1))
        self.menu   = menu

class XVobj:

    nxvs = 0;

    def __init__(self, name=' ', desc=' ', units=' ', 
                 value=0.0*np.ones((1,1)),
                 sstarg=0.0, ssqval=1.0,
                 setpoint=0.0, qvalue=1.0,
                 maxlim=1.0e10, minlim=-1.0e10, roclim=1.0e10,
                 pltmax=100.0, pltmin=0.0,
                 noise=0.0, mnoise=0.000001, dist=0.0, Nf=0, bias=0.0,
                 menu=["value","sstarg","ssqval","setpoint","qvalue",
                       "maxlim","minlim","pltmax","pltmin","mnoise",
                       "noise","dist"]):
        
        XVobj.nxvs += 1

        self.name   = name
        self.desc   = desc
        self.units  = units
        self.value  = value
        self.est    = value
        self.sstarg = sstarg
        self.ssqval = ssqval
        self.setpoint = setpoint
        self.qvalue = qvalue
        self.maxlim = maxlim
        self.minlim = minlim
        self.pltmax = pltmax
        self.pltmin = pltmin
        self.noise  = noise
        self.mnoise = mnoise
        self.dist   = dist
        self.ref    = value
        self.chflag = 0
        self.Nf     = Nf
        self.bias   = bias
        self.olpred = value*np.ones((Nf,1))
        self.clpred = value*np.ones((Nf,1))
        self.menu   = menu

class SimCon:

    def __init__(self, simname=[], 
                 mvlist=[], dvlist=[], cvlist=[], xvlist=[], oplist=[],
                 N=[], refint=[], runsim=[], deltat=[], alg=[], proc=[],
                 mod=[], F=[], l=[], Pf=[], xmk=[], gain=[],
                 ydata=[], udata=[], root=[]):

        self.simname = simname
        self.mvlist  = mvlist
        self.dvlist  = dvlist
        self.cvlist  = cvlist
        self.xvlist  = xvlist
        self.oplist  = oplist
        self.nmvs    = len(mvlist)
        self.ndvs    = len(dvlist)
        self.ncvs    = len(cvlist)
        self.nxvs    = len(xvlist)
        self.xvlist  = xvlist
        self.N       = N
        self.refint  = refint
        self.runsim  = runsim
        self.deltat  = deltat
        self.alg     = alg
        self.proc    = proc
        self.mod     = mod
        self.F       = F
        self.l       = l
        self.Pf      = Pf
        self.xmk     = xmk
        self.gain    = gain
        self.ydata   = ydata
        self.udata   = udata
        self.root    = root



if __name__ == '__main__': execfile('siso_lmpc_mpcsim.py')
