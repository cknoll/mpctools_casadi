import casadi

# this is a workaround for issue #2743:
# it shoud be obsolete for casadi version >= 3.6

# source of this function https://github.com/jcotem/casadi/commit/08c5f39156140817200e207b7696f59223458ce0#diff-3dc05d8ab5896084984b420dd6495df43d2e19e5f7e64693cd516bc2da0dfbf7
# referenced here https://github.com/casadi/casadi/issues/2743

def __array__(self,*args,**kwargs):
    import numpy as n
    if len(args) > 1 and isinstance(args[1],tuple) and isinstance(args[1][0],n.ufunc) and isinstance(args[1][0],n.ufunc) and len(args[1])>1 and args[1][0].nin==len(args[1][1]):
      if len(args[1][1])==3:
        raise Exception("Error with %s. Looks like you are using an assignment operator, such as 'a+=b'. This is not supported when 'a' is a numpy type, and cannot be supported without changing numpy itself. Either upgrade a to a CasADi type first, or use 'a = a + b'. " % args[1][0].__name__)
      return n.array([n.nan])
    else:
      if hasattr(self,'__array_custom__'):
        return self.__array_custom__(*args,**kwargs)
      else:
        try:
          return self.full()
        except:
          if self.is_scalar(True):
            # Needed for #2743
            E=n.empty((),dtype=object)
            E[()] = self
            return E
          else:
            raise Exception("!!Implicit conversion of symbolic CasADi type to numeric matrix not supported.\n"
                      + "This may occur when you pass a CasADi object to a numpy function.\n"
                      + "Use an equivalent CasADi function instead of that numpy function.")


# monkey-patch the casadi datatype

casadi.casadi.SX.__array__ = __array__
