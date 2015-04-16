function rhs = massenbalstst(x)
    global F F0 r E k0 DeltaH rhoCp T0 c0 U Tc hs
    c = x(1);
    T = x(2);
    h = x(3);
    k = k0*exp(-E/T);
    rate = k*c;
    dcdt = F0*(c0-c)/(pi*r^2*h) - rate;
    dTdt = F0*(T0-T)/(pi*r^2*h) - ...
        DeltaH/rhoCp*rate + ...
        2*U/(r*rhoCp)*(Tc-T); 
    % fix the reactor height
    dhdt = h - hs;
    rhs = [dcdt; dTdt; dhdt];
end
