function df = partial(x)
    global F F0 r E k0 DeltaH rhoCp T0 c0 U Tc
    c = x(1);
    T = x(2);
    h = x(3);
    k = k0*exp(-E/T);
    kprime = k*E/(T^2);
    Fprime = F0/(pi*r^2*h);
    df = [
        -Fprime - k, -kprime*c, ...
        -Fprime*(c0-c)/h, ...
        0, 0, (c0-c)/(pi*r^2*h);
        %
        -k*DeltaH/rhoCp, -Fprime ...
            - kprime*c*DeltaH/(rhoCp) ...
            - 2*U/(r*rhoCp), ...
        -Fprime*(T0-T)/h, ...  
        2*U/(r*rhoCp), 0, (T0-T)/(pi*r^2*h);
        %
        0, 0, 0, 0, -1/(pi*r^2), 1/(pi*r^2)
    ];
end
